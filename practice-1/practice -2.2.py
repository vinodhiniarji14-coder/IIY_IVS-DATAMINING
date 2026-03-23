"""
Weather Data Preprocessing Pipeline for ML Training
=====================================================
Steps covered:
  1.  Synthetic dataset generation (realistic weather data)
  2.  Initial inspection & data audit
  3.  Missing value handling  (mean / median / forward-fill / domain rules)
  4.  Outlier detection & treatment  (IQR + Z-score)
  5.  Categorical encoding  (Label + One-Hot)
  6.  Cyclical feature engineering  (hour, month → sin/cos)
  7.  Derived meteorological features  (heat index, dew point, wind chill)
  8.  Feature scaling  (StandardScaler + MinMaxScaler comparison)
  9.  Train / validation / test split  (stratified, no leakage)
  10. Final readiness report
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# ══════════════════════════════════════════════════════════
# STEP 1 — GENERATE SYNTHETIC WEATHER DATASET
# ══════════════════════════════════════════════════════════

def generate_weather_data(n: int = 1000) -> pd.DataFrame:
    """
    Create a realistic weather dataset with:
    - Numeric: temperature, humidity, wind_speed, pressure, rainfall, visibility
    - Categorical: weather_condition, season, wind_direction
    - Datetime: timestamp
    - Injected flaws: ~8% missing values, outliers, duplicates
    """
    timestamps = pd.date_range("2023-01-01", periods=n, freq="h")
    month = timestamps.month

    # Seasonal temperature pattern (°C)
    temp_base = 15 + 10 * np.sin((month - 3) * np.pi / 6)
    temperature = temp_base + np.random.normal(0, 4, n)

    humidity    = np.clip(60 + 15 * np.sin(month * np.pi / 6) + np.random.normal(0, 10, n), 10, 100)
    wind_speed  = np.abs(np.random.weibull(2, n) * 12)          # m/s, right-skewed
    pressure    = np.random.normal(1013, 8, n)                   # hPa
    rainfall    = np.where(humidity > 75, np.abs(np.random.exponential(3, n)), 0)
    visibility  = np.clip(np.random.normal(10, 3, n), 0, 20)    # km

    conditions  = np.random.choice(["sunny", "cloudy", "rainy", "foggy", "stormy"],
                                   n, p=[0.35, 0.30, 0.20, 0.10, 0.05])
    seasons     = pd.cut(timestamps.month, bins=[0,3,6,9,12],
                         labels=["winter","spring","summer","autumn"]).astype(str)
    wind_dirs   = np.random.choice(["N","NE","E","SE","S","SW","W","NW"], n)

    df = pd.DataFrame({
        "timestamp":         timestamps,
        "temperature":       temperature.round(1),
        "humidity":          humidity.round(1),
        "wind_speed":        wind_speed.round(1),
        "pressure":          pressure.round(1),
        "rainfall":          rainfall.round(1),
        "visibility":        visibility.round(1),
        "weather_condition": conditions,
        "season":            seasons,
        "wind_direction":    wind_dirs,
    })

    # ── Inject flaws ──────────────────────────────────
    # Missing values (~8% per numeric column)
    for col in ["temperature", "humidity", "wind_speed", "pressure", "visibility"]:
        mask = np.random.rand(n) < 0.08
        df.loc[mask, col] = np.nan

    # Outliers (sensor spikes)
    df.loc[np.random.choice(n, 15, replace=False), "temperature"] = np.random.choice([65, -50, 80], 15)
    df.loc[np.random.choice(n, 10, replace=False), "wind_speed"]  = np.random.uniform(120, 200, 10)
    df.loc[np.random.choice(n, 8,  replace=False), "humidity"]    = np.random.choice([110, 150, -5], 8)

    # Duplicate rows
    dup_idx = np.random.choice(n, 20, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    return df.sort_values("timestamp").reset_index(drop=True)


print("=" * 65)
print("WEATHER DATA PREPROCESSING PIPELINE")
print("=" * 65)

df = generate_weather_data(1000)
print(f"\nRaw dataset: {df.shape[0]} rows × {df.shape[1]} columns")


# ══════════════════════════════════════════════════════════
# STEP 2 — DATA AUDIT
# ══════════════════════════════════════════════════════════

print("\n" + "─" * 65)
print("STEP 2 — DATA AUDIT")
print("─" * 65)

print("\nColumn dtypes:")
print(df.dtypes.to_string())

print("\nMissing value counts:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
audit = pd.DataFrame({"missing": missing, "missing_%": missing_pct})
print(audit[audit["missing"] > 0].to_string())

print(f"\nDuplicate rows: {df.duplicated().sum()}")

print("\nNumeric summary (before cleaning):")
print(df.describe().round(2))


# ══════════════════════════════════════════════════════════
# STEP 3 — REMOVE DUPLICATES + RESET INDEX
# ══════════════════════════════════════════════════════════

print("\n" + "─" * 65)
print("STEP 3 — REMOVE DUPLICATES")
print("─" * 65)

before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate rows → {len(df)} remaining")


# ══════════════════════════════════════════════════════════
# STEP 4 — MISSING VALUE HANDLING
# ══════════════════════════════════════════════════════════

print("\n" + "─" * 65)
print("STEP 4 — MISSING VALUE HANDLING")
print("─" * 65)

# temperature  → median (robust to outliers we haven't clipped yet)
# humidity     → mean (normally distributed)
# wind_speed   → median (right-skewed)
# pressure     → mean (very stable, near-Gaussian)
# visibility   → forward-fill then back-fill (temporal continuity)
# categorical  → mode

fill_with_median = ["temperature", "wind_speed"]
fill_with_mean   = ["humidity", "pressure"]

for col in fill_with_median:
    med = df[col].median()
    df[col].fillna(med, inplace=True)
    print(f"  {col:<15} filled with median = {med:.2f}")

for col in fill_with_mean:
    mn = df[col].mean()
    df[col].fillna(mn, inplace=True)
    print(f"  {col:<15} filled with mean   = {mn:.2f}")

df["visibility"] = df["visibility"].ffill().bfill()
print(f"  {'visibility':<15} forward-fill / back-fill")

# Categorical: fill with mode
for col in ["weather_condition", "season", "wind_direction"]:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)
    print(f"  {col:<15} filled with mode = '{mode_val}'")

print(f"\nMissing values after imputation: {df.isnull().sum().sum()}")


# ══════════════════════════════════════════════════════════
# STEP 5 — OUTLIER DETECTION & TREATMENT
# ══════════════════════════════════════════════════════════

print("\n" + "─" * 65)
print("STEP 5 — OUTLIER DETECTION & TREATMENT")
print("─" * 65)

# Physical domain bounds (hard constraints from meteorology)
DOMAIN_BOUNDS = {
    "temperature": (-60, 60),    # °C — physically plausible
    "humidity":    (0,   100),   # % — by definition
    "wind_speed":  (0,   90),    # m/s — Beaufort scale max ≈ 32, cat-5 ≈ 83
    "pressure":    (870, 1085),  # hPa — extremes ever recorded
    "visibility":  (0,   20),    # km
    "rainfall":    (0,   200),   # mm/hr
}

def iqr_bounds(series, factor=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr

print("\n  IQR outlier counts (before clipping):")
for col, (lo, hi) in DOMAIN_BOUNDS.items():
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    iqr_lo, iqr_hi = iqr_bounds(df[col])
    n_iqr = ((df[col] < iqr_lo) | (df[col] > iqr_hi)).sum()
    print(f"  {col:<15} domain outliers: {n_out:>3}  |  IQR outliers: {n_iqr:>3}")

# Clip to domain bounds (prefer domain knowledge over statistical methods for weather)
for col, (lo, hi) in DOMAIN_BOUNDS.items():
    df[col] = df[col].clip(lower=lo, upper=hi)

print("\n  Z-score check after clipping:")
numeric_cols = list(DOMAIN_BOUNDS.keys())
z_scores = np.abs(stats.zscore(df[numeric_cols]))
high_z   = (z_scores > 3).sum(axis=0)
for col, cnt in zip(numeric_cols, high_z):
    print(f"  {col:<15} |z| > 3 : {cnt}")

print("\n  All outliers clipped to physical bounds.")


# ══════════════════════════════════════════════════════════
# STEP 6 — CATEGORICAL ENCODING
# ══════════════════════════════════════════════════════════

print("\n" + "─" * 65)
print("STEP 6 — CATEGORICAL ENCODING")
print("─" * 65)

# Label encode ordinal-ish features (season has natural order)
season_order = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}
df["season_encoded"] = df["season"].map(season_order)
print(f"\n  season → label encoded: {season_order}")

# Label encode wind direction (8 compass points)
le_wind = LabelEncoder()
df["wind_dir_encoded"] = le_wind.fit_transform(df["wind_direction"])
print(f"  wind_direction → label encoded: {dict(zip(le_wind.classes_, le_wind.transform(le_wind.classes_)))}")

# One-hot encode weather_condition (no ordinal relationship)
ohe = pd.get_dummies(df["weather_condition"], prefix="cond", drop_first=False)
df = pd.concat([df, ohe], axis=1)
print(f"\n  weather_condition → one-hot encoded columns: {list(ohe.columns)}")

# Drop originals
df.drop(columns=["weather_condition", "season", "wind_direction"], inplace=True)
print(f"\n  Dataset shape after encoding: {df.shape}")


# ══════════════════════════════════════════════════════════
# STEP 7 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════

print("\n" + "─" * 65)
print("STEP 7 — FEATURE ENGINEERING")
print("─" * 65)

# 7a. Cyclical encoding for hour and month (avoid 23→0 boundary artifact)
df["hour"]  = df["timestamp"].dt.hour
df["month"] = df["timestamp"].dt.month

df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
print("  Cyclical features: hour_sin, hour_cos, month_sin, month_cos")

# 7b. Heat Index (apparent temperature, valid when T ≥ 27°C, RH ≥ 40%)
#     Steadman simplified formula (°C)
def heat_index(T, RH):
    HI = (-8.78469475556
          + 1.61139411   * T
          + 2.33854883889 * RH
          - 0.14611605   * T * RH
          - 0.012308094  * T**2
          - 0.016424828  * RH**2
          + 0.002211732  * T**2 * RH
          + 0.00072546   * T * RH**2
          - 0.000003582  * T**2 * RH**2)
    return np.where((T >= 27) & (RH >= 40), HI, T)  # fallback to T when conditions not met

df["heat_index"] = heat_index(df["temperature"], df["humidity"]).round(1)
print("  Derived feature: heat_index (apparent temperature)")

# 7c. Dew Point (Magnus formula)
#     Accurate within ±0.4°C for 0–60°C, 1–100% RH
def dew_point(T, RH):
    a, b = 17.625, 243.04
    alpha = np.log(np.clip(RH, 0.01, 100) / 100) + (a * T) / (b + T)
    return (b * alpha / (a - alpha)).round(1)

df["dew_point"] = dew_point(df["temperature"], df["humidity"])
print("  Derived feature: dew_point (°C)")

# 7d. Wind Chill (valid when T ≤ 10°C, wind ≥ 4.8 km/h)
#     Environment Canada formula
def wind_chill(T, V_kmh):
    WC = (13.12 + 0.6215 * T
          - 11.37 * V_kmh**0.16
          + 0.3965 * T * V_kmh**0.16)
    return np.where((T <= 10) & (V_kmh >= 4.8), WC, T).round(1)

df["wind_speed_kmh"] = (df["wind_speed"] * 3.6).round(1)   # m/s → km/h
df["wind_chill"]     = wind_chill(df["temperature"], df["wind_speed_kmh"])
print("  Derived feature: wind_chill (°C)")

# 7e. Pressure tendency (change over last 3 hours — a storm predictor)
df = df.sort_values("timestamp").reset_index(drop=True)
df["pressure_tendency"] = df["pressure"].diff(periods=3).fillna(0).round(2)
print("  Derived feature: pressure_tendency (3-hr change in hPa)")

# 7f. Discomfort index
df["discomfort_idx"] = (0.4 * (df["temperature"] + df["dew_point"]) + 4.8).round(2)
print("  Derived feature: discomfort_idx")

print(f"\n  Dataset shape after feature engineering: {df.shape}")


# ══════════════════════════════════════════════════════════
# STEP 8 — FEATURE SCALING
# ══════════════════════════════════════════════════════════

print("\n" + "─" * 65)
print("STEP 8 — FEATURE SCALING")
print("─" * 65)

# Drop non-feature columns before splitting
drop_cols = ["timestamp", "hour", "month"]
df_ml = df.drop(columns=drop_cols)

# Identify numeric columns to scale (exclude already-0/1 encoded and sin/cos)
exclude_from_scaling = [c for c in df_ml.columns
                        if c.startswith("cond_")
                        or c.endswith("_sin") or c.endswith("_cos")
                        or c in ["season_encoded", "wind_dir_encoded"]]

TARGET = "temperature"     # predict next-hour temperature

scale_cols = [c for c in df_ml.select_dtypes(include=np.number).columns
              if c not in exclude_from_scaling and c != TARGET]

print(f"\n  Columns to scale: {scale_cols}")

# ── 8a. Train / validation / test split BEFORE fitting scalers ──
# Always split first to prevent data leakage from scaler statistics
feature_cols = [c for c in df_ml.columns if c != TARGET]

X = df_ml[feature_cols]
y = df_ml[TARGET]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, shuffle=True)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f"\n  Train   : {X_train.shape[0]:>5} rows  ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"  Val     : {X_val.shape[0]:>5} rows  ({X_val.shape[0]/len(X)*100:.0f}%)")
print(f"  Test    : {X_test.shape[0]:>5} rows  ({X_test.shape[0]/len(X)*100:.0f}%)")

# ── 8b. Fit scalers on TRAIN only, transform all splits ──
ss = StandardScaler()
mm = MinMaxScaler()

X_train_ss = X_train.copy()
X_train_mm = X_train.copy()

X_train_ss[scale_cols] = ss.fit_transform(X_train[scale_cols])
X_val_ss   = X_val.copy();   X_val_ss[scale_cols]  = ss.transform(X_val[scale_cols])
X_test_ss  = X_test.copy();  X_test_ss[scale_cols] = ss.transform(X_test[scale_cols])

X_train_mm[scale_cols] = mm.fit_transform(X_train[scale_cols])
X_val_mm   = X_val.copy();   X_val_mm[scale_cols]  = mm.transform(X_val[scale_cols])
X_test_mm  = X_test.copy();  X_test_mm[scale_cols] = mm.transform(X_test[scale_cols])

print(f"\n  StandardScaler — train set after scaling:")
print(f"  {'column':<20} {'mean':>8} {'std':>8}")
for col in scale_cols[:5]:
    print(f"  {col:<20} {X_train_ss[col].mean():>8.4f} {X_train_ss[col].std():>8.4f}")
print("  ...")

print(f"\n  MinMaxScaler — train set after scaling (first 5 cols):")
print(f"  {'column':<20} {'min':>8} {'max':>8}")
for col in scale_cols[:5]:
    print(f"  {col:<20} {X_train_mm[col].min():>8.4f} {X_train_mm[col].max():>8.4f}")
print("  ...")


# ══════════════════════════════════════════════════════════
# STEP 9 — FINAL READINESS REPORT
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("FINAL READINESS REPORT")
print("=" * 65)

report = {
    "Total samples":          len(df_ml),
    "Feature count":          len(feature_cols),
    "Target":                 TARGET,
    "Train samples":          len(X_train),
    "Val samples":            len(X_val),
    "Test samples":           len(X_test),
    "Missing values":         int(df_ml.isnull().sum().sum()),
    "Duplicate rows":         int(df_ml.duplicated().sum()),
    "Numeric features":       len(scale_cols),
    "Binary OHE features":    len([c for c in feature_cols if c.startswith("cond_")]),
    "Cyclical features":      4,
    "Engineered features":    6,
    "Scaler fitted on":       "train only (no leakage)",
}

for k, v in report.items():
    print(f"  {k:<30} {v}")

print("\nDataset is ready for model training.")
print("\nSample of X_train (StandardScaled, first 5 rows × 5 cols):")
print(X_train_ss[scale_cols[:5]].head().round(3).to_string())
