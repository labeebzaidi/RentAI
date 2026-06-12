# ==========================================================
# HOUSE RENT PRICE PREDICTION
# Phase 1 : Data Loading + Cleaning + Preprocessing
# ==========================================================

import pandas as pd
import numpy as np

# -----------------------------
# Load Dataset
# -----------------------------

# Change filename if required
df = pd.read_csv("Makaan_data_700pages.csv")

print("=" * 60)
print("ORIGINAL SHAPE")
print("=" * 60)
print(df.shape)

# -----------------------------
# Drop unnecessary columns
# -----------------------------

drop_cols = []

for col in [
    "Unnamed: 0",
    "Seller_name",
    "Facing_direction"
]:
    if col in df.columns:
        drop_cols.append(col)

df = df.drop(columns=drop_cols)

print("\nDropped Columns:")
print(drop_cols)

# -----------------------------
# Clean Rent Price
# -----------------------------

def convert_money(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    value = value.replace(",", "")

    if value.lower() == "no":
        return 0

    if "L" in value:
        try:
            value = value.replace("L", "").strip()
            return float(value) * 100000
        except:
            return np.nan

    try:
        return float(value)
    except:
        return np.nan


df["Rent_price"] = df["Rent_price"].apply(convert_money)

# -----------------------------
# Clean Area_sqft
# -----------------------------

df["Area_sqft"] = (
    df["Area_sqft"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["Area_sqft"] = pd.to_numeric(
    df["Area_sqft"],
    errors="coerce"
)

# -----------------------------
# Bathroom
# -----------------------------

df["Bathroom"] = pd.to_numeric(
    df["Bathroom"],
    errors="coerce"
)

df["Bathroom"] = df["Bathroom"].fillna(
    df["Bathroom"].median()
)

# -----------------------------
# Remove rows where target missing
# -----------------------------

df = df.dropna(subset=["Rent_price"])

# -----------------------------
# Fill categorical missing
# -----------------------------

categorical_cols = [
    "Size_unit",
    "Property_type",
    "Location",
    "Seller_type",
    "Status"
]

for col in categorical_cols:

    if col in df.columns:

        df[col] = df[col].fillna(
            df[col].mode()[0]
        )

# -----------------------------
# Fill numeric missing
# -----------------------------

numeric_cols = [
    "Size",
    "Area_sqft"
]

for col in numeric_cols:

    if col in df.columns:

        df[col] = df[col].fillna(
            df[col].median()
        )

# -----------------------------
# Select Features
# -----------------------------

features = [
    "Size",
    "Size_unit",
    "Property_type",
    "Location",
    "Seller_type",
    "Area_sqft",
    "Status",
    "Bathroom"
]

X = df[features]

y = df["Rent_price"]

# -----------------------------
# One Hot Encoding
# -----------------------------

X = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)

print("\n")
print("=" * 60)
print("PREPROCESSED DATA")
print("=" * 60)

print("Feature Shape :", X.shape)
print("Target Shape  :", y.shape)

print("\nFirst 5 Encoded Rows")
print(X.head())

print("\nTarget Sample")
print(y.head())



# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n")
print("="*60)
print("TRAIN TEST SPLIT")
print("="*60)

print("Training Samples :",len(X_train))
print("Testing Samples  :",len(X_test))

# ==========================================================
# RANDOM FOREST MODEL
# ==========================================================

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(

    n_estimators=300,

    random_state=42,

    n_jobs=-1,

    max_depth=20,

    min_samples_split=5,

    min_samples_leaf=2

)

print("\nTraining Model...")

model.fit(
    X_train,
    y_train
)

print("Model Trained Successfully ✅")


# ==========================================================
# PREDICTION
# ==========================================================

y_pred = model.predict(X_test)

print("\nPrediction Completed ✅")

# ==========================================================
# EVALUATION
# ==========================================================

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

import numpy as np

print("\n")
print("="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(
    "R2 Score :",
    round(
        r2_score(
            y_test,
            y_pred
        ),
        4
    )
)

print(
    "MAE :",
    round(
        mean_absolute_error(
            y_test,
            y_pred
        ),
        2
    )
)

print(
    "RMSE :",
    round(
        np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        ),
        2
    )
)



import matplotlib.pyplot as plt

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(
    ascending=False
)

top20 = importance.head(20)

plt.figure(figsize=(10,8))

top20.sort_values().plot(kind="barh")

plt.title("Top 20 Important Features")

plt.xlabel("Importance Score")

plt.tight_layout()

plt.savefig("Feature_Importance.png")

plt.show()





# plt.figure(figsize=(7,7))

# plt.scatter(
#     y_test,
#     y_pred,
#     alpha=0.5
# )

# plt.xlabel("Actual Rent")

# plt.ylabel("Predicted Rent")

# plt.title("Actual vs Predicted Rent")

# min_val = min(y_test.min(), y_pred.min())
# max_val = max(y_test.max(), y_pred.max())

# plt.plot(
#     [min_val, max_val],
#     [min_val, max_val]
# )

# plt.tight_layout()

# plt.savefig("Actual_vs_Predicted.png")

# plt.show()


plot_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

plot_df = plot_df.sample(800, random_state=42)

plt.figure(figsize=(8,6))

plt.scatter(
    plot_df["Actual"],
    plot_df["Predicted"],
    alpha=0.35,
    s=20
)

min_val = min(plot_df["Actual"].min(), plot_df["Predicted"].min())
max_val = max(plot_df["Actual"].max(), plot_df["Predicted"].max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)

plt.xlabel("Actual Rent", fontsize=12)
plt.ylabel("Predicted Rent", fontsize=12)
plt.title("Actual vs Predicted Rent", fontsize=15, fontweight="bold")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    "Actual_vs_Predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()




pred_df = pd.DataFrame({

    "Actual Rent": y_test.values,

    "Predicted Rent": y_pred

})

pred_df.to_csv(
    "Predictions.csv",
    index=False
)

print("\nPredictions.csv saved successfully ✅")



# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

# Sort by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print("="*60)
print("TOP 15 IMPORTANT FEATURES")
print("="*60)

print(importance_df.head(15))

# Take top 15 features
top15 = importance_df.head(15)

# Plot
plt.figure(figsize=(10,6))

plt.barh(
    top15["Feature"][::-1],
    top15["Importance"][::-1]
)

plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Top 15 Most Important Features")

plt.grid(axis="x", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "Feature_Importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



import joblib

joblib.dump(model, "model.pkl")

joblib.dump(X.columns.tolist(), "encoder_columns.pkl")

print("\nModel saved successfully ✅")