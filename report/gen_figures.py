"""Reproduce key analyses from the repo's notebooks and emit figures + metrics
for the ACM report.  Saves PDFs under the report/figures/ directory and a JSON
file with numeric results for embedding in the LaTeX text.
"""
from __future__ import annotations
import json
import os
import re
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
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
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
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

metrics: dict = {}

# ---------------------------------------------------------------------------
# 1. Vietnam dataset analysis (replicating BDSVietNam.ipynb)
# ---------------------------------------------------------------------------
print("[1/3] Vietnam dataset ...")
vn = pd.read_csv(os.path.join(REPO, "BDSVietNam", "batdongsan_data.csv"))


def parse_numeric(text):
    if pd.isna(text) or "Thỏa thuận" in str(text):
        return np.nan
    m = re.search(r"(\d+[.,]?\d*)", str(text).replace(",", "."))
    return float(m.group(1)) if m else np.nan


districts = [
    "Quận 1", "Quận 2", "Quận 3", "Quận 4", "Quận 5", "Quận 6", "Quận 7",
    "Quận 8", "Quận 9", "Quận 10", "Quận 11", "Quận 12",
    "Tân Bình", "Bình Tân", "Tân Phú", "Gò Vấp", "Phú Nhuận",
    "Bình Thạnh", "Thủ Đức", "Cầu Giấy", "Đống Đa", "Hà Đông",
]


def parse_district(text):
    text = str(text).lower()
    for d in districts:
        if d.lower() in text:
            return d
    return "Khác"


vn["Price_Clean"] = vn["Price"].apply(parse_numeric)
vn["Price_Clean"] = np.where(
    vn["Price"].astype(str).str.contains("tỷ", na=False),
    vn["Price_Clean"] * 1000,
    vn["Price_Clean"],
)
vn["Area_Clean"] = vn["Area"].apply(parse_numeric)
vn["District"] = (vn["Title"].fillna("") + " " + vn["Description"].fillna("")) \
    .apply(parse_district)

n_raw = len(vn)
vn = vn.dropna(subset=["Price_Clean", "Area_Clean"])
vn = vn[(vn["Price_Clean"] > 200) & (vn["Area_Clean"] > 15)]
n_clean = len(vn)
metrics["vn_raw_rows"] = int(n_raw)
metrics["vn_clean_rows"] = int(n_clean)
metrics["vn_n_districts"] = int(vn["District"].nunique())

le = LabelEncoder()
vn["District_Enc"] = le.fit_transform(vn["District"])

# Distribution figure
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
sns.histplot(vn["Price_Clean"], bins=40, ax=axes[0], color="#2a7a8c")
axes[0].set_xlabel("Giá (triệu VND)")
axes[0].set_ylabel("Số tin đăng")
axes[0].set_title("(a) Phân phối giá")
axes[0].set_xlim(0, vn["Price_Clean"].quantile(0.98))

sns.histplot(vn["Area_Clean"], bins=40, ax=axes[1], color="#a05a2c")
axes[1].set_xlabel("Diện tích (m²)")
axes[1].set_ylabel("Số tin đăng")
axes[1].set_title("(b) Phân phối diện tích")
axes[1].set_xlim(0, vn["Area_Clean"].quantile(0.98))

top_dist = vn["District"].value_counts().head(10).index
sub = vn[vn["District"].isin(top_dist)]
order = (
    sub.groupby("District")["Price_Clean"].median().sort_values().index
)
sns.boxplot(
    data=sub, x="Price_Clean", y="District", order=order, ax=axes[2],
    color="#5a8c5a", showfliers=False,
)
axes[2].set_xlabel("Giá (triệu VND)")
axes[2].set_ylabel("")
axes[2].set_title("(c) Giá theo top 10 quận")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "vn_distributions.pdf"))
plt.close()

# Train Stacking ensemble
X = vn[["Area_Clean", "District_Enc"]]
y = np.log1p(vn["Price_Clean"])
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)
base_models = [
    ("xgb", xgb.XGBRegressor(n_estimators=500, max_depth=6,
                             learning_rate=0.03, random_state=42,
                             verbosity=0)),
    ("lgb", lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03,
                              num_leaves=31, verbose=-1, random_state=42)),
    ("cat", CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6,
                              verbose=0, random_state=42)),
]
stack = StackingRegressor(estimators=base_models,
                          final_estimator=Ridge(alpha=1.0))
stack.fit(X_tr, y_tr)
y_pr = stack.predict(X_te)
vn_r2 = float(r2_score(y_te, y_pr))
vn_mae = float(mean_absolute_error(np.expm1(y_te), np.expm1(y_pr)))
vn_rmse_log = float(np.sqrt(mean_squared_error(y_te, y_pr)))
metrics["vn_stacking_r2"] = round(vn_r2, 4)
metrics["vn_stacking_mae_million_vnd"] = round(vn_mae, 1)
metrics["vn_stacking_rmse_log"] = round(vn_rmse_log, 4)

fig, ax = plt.subplots(figsize=(5, 4.2))
sns.regplot(
    x=np.expm1(y_te), y=np.expm1(y_pr),
    scatter_kws={"alpha": 0.45, "color": "#1f77b4", "s": 18},
    line_kws={"color": "#d62728"}, ax=ax,
)
mx = max(np.expm1(y_te).max(), np.expm1(y_pr).max())
ax.plot([0, mx], [0, mx], ls="--", c="grey", lw=0.8, label="y = x")
ax.set_xlabel("Giá thực tế (triệu VND)")
ax.set_ylabel("Giá dự báo (triệu VND)")
ax.set_title(f"VN Stacking — $R^2={vn_r2:.3f}$, MAE={vn_mae:,.0f} triệu")
ax.legend(loc="upper left")
ax.set_xlim(0, mx * 1.02)
ax.set_ylim(0, mx * 1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "vn_stacking_pred_vs_true.pdf"))
plt.close()

# ---------------------------------------------------------------------------
# 2. Kaggle Boosting comparison (replicating Boosting.ipynb)
# ---------------------------------------------------------------------------
print("[2/3] Kaggle Boosting comparison ...")
train = pd.read_csv(os.path.join(REPO, "Kaggle", "train.csv"))
y_kag = np.log1p(train["SalePrice"])
X_kag = train.drop(["SalePrice", "Id"], axis=1)
for col in X_kag.columns:
    if X_kag[col].dtype == "object" or pd.api.types.is_string_dtype(X_kag[col]):
        X_kag[col] = LabelEncoder().fit_transform(X_kag[col].astype(str))
    else:
        X_kag[col] = X_kag[col].fillna(X_kag[col].median())
X_tr, X_te, y_tr, y_te = train_test_split(
    X_kag, y_kag, test_size=0.2, random_state=42
)

models = {
    "GBM": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                     max_depth=4, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=500, learning_rate=0.05,
                                max_depth=6, random_state=42, verbosity=0),
    "LightGBM": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                  verbose=-1, random_state=42),
    "CatBoost": CatBoostRegressor(iterations=500, learning_rate=0.05,
                                  depth=6, verbose=0, random_state=42),
}
fitted = {}
boost_rows = []
for name, mdl in models.items():
    t0 = time.time()
    mdl.fit(X_tr, y_tr)
    dt = time.time() - t0
    pr = mdl.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, pr)))
    r2 = float(r2_score(y_te, pr))
    mae = float(mean_absolute_error(np.expm1(y_te), np.expm1(pr)))
    boost_rows.append({"Model": name, "RMSE": rmse, "R2": r2,
                       "MAE": mae, "Time (s)": dt})
    fitted[name] = mdl

ensemble = VotingRegressor(estimators=[(n.lower(), fitted[n])
                                       for n in models])
t0 = time.time()
ensemble.fit(X_tr, y_tr)
dt = time.time() - t0
pr = ensemble.predict(X_te)
rmse = float(np.sqrt(mean_squared_error(y_te, pr)))
r2 = float(r2_score(y_te, pr))
mae = float(mean_absolute_error(np.expm1(y_te), np.expm1(pr)))
boost_rows.append({"Model": "Voting Ensemble", "RMSE": rmse, "R2": r2,
                   "MAE": mae, "Time (s)": dt})

boost_df = pd.DataFrame(boost_rows)
boost_df.to_csv(os.path.join(OUT, "boosting_results.csv"), index=False)
metrics["boosting"] = {
    r["Model"]: {"rmse_log": round(r["RMSE"], 4),
                 "r2": round(r["R2"], 4),
                 "mae_usd": round(r["MAE"], 1),
                 "time_s": round(r["Time (s)"], 2)}
    for r in boost_rows
}

# Bar chart RMSE & training time
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
order_rmse = boost_df.sort_values("RMSE")
sns.barplot(data=order_rmse, x="Model", y="RMSE", ax=axes[0],
            palette="viridis")
axes[0].set_title("(a) RMSE trên log(SalePrice) — thấp hơn là tốt hơn")
axes[0].set_xlabel("")
axes[0].set_ylabel("RMSE")
axes[0].tick_params(axis="x", rotation=20)
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height():.4f}",
                     (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha="center", va="bottom", fontsize=8)

sns.barplot(data=boost_df, x="Model", y="Time (s)", ax=axes[1],
            palette="magma")
axes[1].set_title("(b) Thời gian huấn luyện (giây)")
axes[1].set_xlabel("")
axes[1].set_ylabel("Thời gian (s)")
axes[1].tick_params(axis="x", rotation=20)
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height():.1f}",
                     (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "kaggle_boosting_comparison.pdf"))
plt.close()

# Feature importance comparison (Top-10 from CatBoost & XGBoost & LightGBM)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (name, palette) in zip(
        axes, [("CatBoost", "Blues_d"), ("XGBoost", "Oranges_d"),
               ("LightGBM", "Greens_d")]):
    mdl = fitted[name]
    if name == "CatBoost":
        imp = mdl.get_feature_importance()
    else:
        imp = mdl.feature_importances_
    fi = pd.DataFrame({"Feature": X_kag.columns, "Importance": imp})
    fi = fi.sort_values("Importance", ascending=False).head(10)
    sns.barplot(data=fi, x="Importance", y="Feature", ax=ax, palette=palette)
    ax.set_title(f"{name}: Top-10 đặc trưng")
    ax.set_xlabel("Tầm quan trọng")
    ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "kaggle_feature_importance.pdf"))
plt.close()

# Distribution before/after log transform
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
sns.histplot(train["SalePrice"], bins=40, ax=axes[0], color="#a05a2c")
axes[0].set_title(f"(a) SalePrice (skew={skew(train['SalePrice']):.2f})")
axes[0].set_xlabel("USD")
sns.histplot(np.log1p(train["SalePrice"]), bins=40, ax=axes[1],
             color="#2a7a8c")
axes[1].set_title(
    f"(b) log(1+SalePrice) (skew={skew(np.log1p(train['SalePrice'])):.2f})")
axes[1].set_xlabel("log(USD)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "kaggle_log_transform.pdf"))
plt.close()

# ---------------------------------------------------------------------------
# 3. Save metrics.json
# ---------------------------------------------------------------------------
metrics["dataset_kaggle_train_rows"] = int(len(train))
metrics["dataset_kaggle_test_rows"] = int(
    len(pd.read_csv(os.path.join(REPO, "Kaggle", "test.csv"))))
metrics["dataset_kaggle_features"] = int(X_kag.shape[1])

with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("Done. Metrics:")
print(json.dumps(metrics, indent=2, ensure_ascii=False))
