#!/usr/bin/env python3
"""Step 3 - Baseline fraud detector on IEEE-CIS. Time-ordered split."""

import os
import gc
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = "data"
OUT = "artifacts"
TEST_FRACTION = 0.20
SEED = 42

os.makedirs(OUT, exist_ok=True)


def reduce_mem(df):
    for col in df.columns:
        dt = df[col].dtype
        if dt == "float64":
            df[col] = df[col].astype("float32")
        elif dt == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


print("Loading transaction file...")
tx = pd.read_csv(f"{DATA}/train_transaction.csv")
print(f"  transactions: {tx.shape}")

print("Loading identity file...")
idf = pd.read_csv(f"{DATA}/train_identity.csv")
print(f"  identity:     {idf.shape}")

df = tx.merge(idf, on="TransactionID", how="left")
del tx, idf
gc.collect()

df = reduce_mem(df)
print(f"\nMerged: {df.shape}")
print(f"Fraud rate: {df['isFraud'].mean():.4%}")
print(f"Fraud count: {int(df['isFraud'].sum()):,} of {len(df):,}")

df = df.sort_values("TransactionDT").reset_index(drop=True)

split_at = int(len(df) * (1 - TEST_FRACTION))
split_dt = df.loc[split_at, "TransactionDT"]

train_df = df.iloc[:split_at].copy()
test_df = df.iloc[split_at:].copy()

print(f"\nSplit at TransactionDT = {split_dt:,}")
print(f"  train: {len(train_df):,} rows, fraud {train_df['isFraud'].mean():.4%}")
print(f"  test:  {len(test_df):,} rows, fraud {test_df['isFraud'].mean():.4%}")

del df
gc.collect()

DROP = ["TransactionID", "isFraud", "TransactionDT"]
feature_cols = [c for c in train_df.columns if c not in DROP]

cat_cols = [c for c in feature_cols if train_df[c].dtype == "object"]
print(f"\nFeatures: {len(feature_cols)}  (categorical: {len(cat_cols)})")

for col in cat_cols:
    combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    codes, _ = pd.factorize(combined)
    train_df[col] = codes[: len(train_df)]
    test_df[col] = codes[len(train_df) :]
    train_df[col] = train_df[col].astype("int32")
    test_df[col] = test_df[col].astype("int32")

X_train = train_df[feature_cols]
y_train = train_df["isFraud"]
X_test = test_df[feature_cols]
y_test = test_df["isFraud"]

params = {
    "objective": "binary",
    "metric": ["auc", "average_precision"],
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 128,
    "min_child_samples": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "is_unbalance": True,
    "verbosity": -1,
    "seed": SEED,
    "num_threads": -1,
}

dtrain = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)
dtest = lgb.Dataset(X_test, y_test, categorical_feature=cat_cols, reference=dtrain)

print("\nTraining...")
model = lgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    valid_sets=[dtest],
    valid_names=["holdout"],
    callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(100)],
)

pred = model.predict(X_test, num_iteration=model.best_iteration)

auc = roc_auc_score(y_test, pred)
pr_auc = average_precision_score(y_test, pred)

print("\n" + "=" * 46)
print("HELD-OUT RESULTS (last 20% by time)")
print("=" * 46)
print(f"  ROC-AUC        : {auc:.4f}")
print(f"  PR-AUC         : {pr_auc:.4f}")
print(f"  Baseline PR-AUC: {y_test.mean():.4f}")
print(f"  Best iteration : {model.best_iteration}")
print("=" * 46)

model.save_model(f"{OUT}/model.txt", num_iteration=model.best_iteration)

holdout = pd.DataFrame(
    {
        "TransactionID": test_df["TransactionID"].values,
        "TransactionDT": test_df["TransactionDT"].values,
        "TransactionAmt": test_df["TransactionAmt"].values,
        "isFraud": y_test.values,
        "pred": pred,
    }
)
holdout.to_csv(f"{OUT}/holdout_predictions.csv", index=False)

X_test.to_parquet(f"{OUT}/X_test.parquet", index=False)

imp = pd.DataFrame(
    {"feature": model.feature_name(), "gain": model.feature_importance("gain")}
).sort_values("gain", ascending=False)
imp.to_csv(f"{OUT}/feature_importance.csv", index=False)

with open(f"{OUT}/metrics.json", "w") as f:
    json.dump(
        {
            "roc_auc": float(auc),
            "pr_auc": float(pr_auc),
            "test_fraud_rate": float(y_test.mean()),
            "train_fraud_rate": float(y_train.mean()),
            "best_iteration": int(model.best_iteration),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_features": int(len(feature_cols)),
            "split_transaction_dt": int(split_dt),
        },
        f,
        indent=2,
    )

print("\nTop 15 features by gain:")
print(imp.head(15).to_string(index=False))

print(f"\nSaved to {OUT}/")
