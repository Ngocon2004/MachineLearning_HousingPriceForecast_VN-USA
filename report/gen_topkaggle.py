"""Reproduce the blended Stacking solution (2212363_DGHaHai.ipynb) on a holdout
split of the Kaggle training set so we can report a real validation RMSE/R2
in the ACM report."""
from __future__ import annotations
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import skew
from scipy.special import boxcox1p
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

REPO = "/home/ubuntu/repos/MachineLearning_HousingPriceForecast_VN-USA"
OUT = os.path.join(REPO, "report", "figures")
os.makedirs(OUT, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

# ----- Load and split full train.csv (so we have ground truth for holdout) -----
train_full = pd.read_csv(os.path.join(REPO, "Kaggle", "train.csv"))
# Outlier removal as in the notebook
train_full = train_full.drop(
    train_full[(train_full["GrLivArea"] > 4000)
               & (train_full["SalePrice"] < 300000)].index
).reset_index(drop=True)
y_full = np.log1p(train_full["SalePrice"].values)
features = train_full.drop(["SalePrice", "Id"], axis=1)

# Apply the FE pipeline once on the full set (same statistics for both
# training and holdout — matches the original notebook's approach which fits
# everything on the train+test concatenation).
all_data = features.copy()

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"] \
    .transform(lambda s: s.fillna(s.median()))
for c in ("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
          "GarageType", "GarageFinish", "GarageQual", "GarageCond",
          "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
          "BsmtFinType2", "MasVnrType"):
    all_data[c] = all_data[c].fillna("None")
for c in ("GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1",
          "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath",
          "BsmtHalfBath", "MasVnrArea"):
    all_data[c] = all_data[c].fillna(0)
all_data["MSZoning"] = all_data["MSZoning"].fillna(
    all_data["MSZoning"].mode()[0])
all_data = all_data.drop(["Utilities"], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
for c in ("Electrical", "KitchenQual", "Exterior1st",
          "Exterior2nd", "SaleType"):
    all_data[c] = all_data[c].fillna(all_data[c].mode()[0])

all_data["TotalSF"] = all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + \
    all_data["2ndFlrSF"]
all_data["TotalBath"] = (all_data["FullBath"]
                         + 0.5 * all_data["HalfBath"]
                         + all_data["BsmtFullBath"]
                         + 0.5 * all_data["BsmtHalfBath"])
all_data["HouseAge"] = all_data["YrSold"] - all_data["YearBuilt"]
all_data["RemodAge"] = all_data["YrSold"] - all_data["YearRemodAdd"]
all_data["IsNew"] = (all_data["YearBuilt"] == all_data["YrSold"]).astype(int)

all_data["MSSubClass"] = all_data["MSSubClass"].apply(str)
cols_ord = ("FireplaceQu", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond",
            "ExterQual", "ExterCond", "HeatingQC", "PoolQC", "KitchenQual",
            "BsmtFinType1", "BsmtFinType2", "Functional", "Fence",
            "BsmtExposure", "GarageFinish", "LandSlope", "LotShape",
            "PavedDrive", "Street", "Alley", "CentralAir")
for c in cols_ord:
    all_data[c] = LabelEncoder().fit_transform(list(all_data[c].values))

numeric_feats = [c for c in all_data.columns
                 if pd.api.types.is_numeric_dtype(all_data[c])]
skewed = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) \
    .sort_values(ascending=False)
skewness = pd.DataFrame({"Skew": skewed})
skewness = skewness[abs(skewness) > 0.75]
for feat in skewness.index:
    all_data[feat] = boxcox1p(all_data[feat], 0.15)

all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.median(numeric_only=True))

X_arr = all_data.values
print("Feature matrix shape:", X_arr.shape)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_arr, y_full, test_size=0.2, random_state=42)

# Configure base learners (same hyperparameters as notebook)
kf = KFold(5, shuffle=True, random_state=42)
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
enet = make_pipeline(RobustScaler(),
                     ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3))
krr = Ridge(alpha=0.6)
svr = make_pipeline(RobustScaler(),
                    SVR(C=20, epsilon=0.008, gamma=0.0003))
gbm = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                max_depth=4, max_features="sqrt",
                                min_samples_leaf=15, min_samples_split=10,
                                loss="huber", random_state=5)
mxgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                        learning_rate=0.05, max_depth=3,
                        min_child_weight=1.7817, n_estimators=2200,
                        reg_alpha=0.4640, reg_lambda=0.8571,
                        subsample=0.5213, random_state=7, verbosity=0)
mlgb = lgb.LGBMRegressor(objective="regression", num_leaves=5,
                         learning_rate=0.05, n_estimators=720, max_bin=55,
                         bagging_fraction=0.8, bagging_freq=5,
                         feature_fraction=0.2319, verbose=-1,
                         random_state=42)
mcat = CatBoostRegressor(iterations=2000, learning_rate=0.01, depth=4,
                         l2_leaf_reg=3, loss_function="RMSE", verbose=0,
                         random_state=42)
stack_gen = StackingRegressor(
    estimators=[("lasso", lasso), ("enet", enet), ("krr", krr),
                ("svr", svr), ("gbm", gbm)],
    final_estimator=lasso, cv=kf,
)

print("Fitting stack_gen ...")
t0 = time.time()
stack_gen.fit(X_tr, y_tr)
print(f"  done in {time.time() - t0:.1f}s")
print("Fitting xgb / lgb / cat ...")
t0 = time.time()
mxgb.fit(X_tr, y_tr)
mlgb.fit(X_tr, y_tr)
mcat.fit(X_tr, y_tr)
print(f"  done in {time.time() - t0:.1f}s")


def blend_pred(X):
    return (0.40 * stack_gen.predict(X)
            + 0.20 * mxgb.predict(X)
            + 0.20 * mlgb.predict(X)
            + 0.20 * mcat.predict(X))


# Per-component holdout metrics
results = {}
for name, mdl in [("Stacked-Linear", stack_gen),
                  ("XGBoost (tuned)", mxgb),
                  ("LightGBM (tuned)", mlgb),
                  ("CatBoost (tuned)", mcat)]:
    pr = mdl.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, pr)))
    r2 = float(r2_score(y_te, pr))
    mae = float(mean_absolute_error(np.expm1(y_te), np.expm1(pr)))
    results[name] = {"rmse_log": round(rmse, 4),
                     "r2": round(r2, 4),
                     "mae_usd": round(mae, 1)}

pr = blend_pred(X_te)
rmse = float(np.sqrt(mean_squared_error(y_te, pr)))
r2 = float(r2_score(y_te, pr))
mae = float(mean_absolute_error(np.expm1(y_te), np.expm1(pr)))
results["Blended (40/20/20/20)"] = {"rmse_log": round(rmse, 4),
                                    "r2": round(r2, 4),
                                    "mae_usd": round(mae, 1)}

for n, r in results.items():
    print(f"{n:25s}  RMSE={r['rmse_log']:.4f}  R2={r['r2']:.4f}  "
          f"MAE=${r['mae_usd']:,.0f}")

# Save metrics
out_path = os.path.join(OUT, "metrics_topkaggle.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

# Plot: actual vs predicted for blended model
fig, ax = plt.subplots(figsize=(5.0, 4.4))
sns.scatterplot(x=np.expm1(y_te), y=np.expm1(pr), alpha=0.55,
                color="#1f77b4", s=22, ax=ax)
mx = max(np.expm1(y_te).max(), np.expm1(pr).max())
ax.plot([0, mx], [0, mx], ls="--", color="#d62728", lw=1, label="y = x")
ax.set_xlabel("SalePrice thực tế (USD)")
ax.set_ylabel("SalePrice dự báo (USD)")
ax.set_title(
    f"US Blended Stacking — $R^2$={r2:.3f}, RMSE(log)={rmse:.4f}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "kaggle_blended_pred_vs_true.pdf"))
plt.close()

print("Saved", out_path)
