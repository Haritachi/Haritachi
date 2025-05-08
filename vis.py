# eda_telco_churn.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set(style="whitegrid")

# Load dataset
df = pd.read_csv("churn2.csv")

# ---- 1. Basic Info ----
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFirst 5 rows:")
print(df.head())

# ---- 2. Handling 'TotalCharges' Missing/Invalid Data ----
print("\nBlank 'TotalCharges':", len(df[df["TotalCharges"] == " "]))
df["TotalCharges"] = df["TotalCharges"].replace(" ", "0.0").astype(float)

# ---- 3. Summary Statistics ----
print("\nStatistical Summary:")
print(df.describe())

# ---- 4. Numerical Feature Distributions ----
numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
for col in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, color="skyblue")
    plt.axvline(df[col].mean(), color="red", linestyle="--", label="Mean")
    plt.axvline(df[col].median(), color="green", label="Median")
    plt.title(f"Distribution of {col}")
    plt.legend()
    plt.show()

# ---- 5. Boxplots (Outlier Detection) ----
for col in numerical_features:
    plt.figure(figsize=(4, 3))
    sns.boxplot(y=df[col], color="orange")
    plt.title(f"Boxplot of {col}")
    plt.show()

# ---- 6. Correlation Heatmap ----
plt.figure(figsize=(6, 4))
sns.heatmap(df[numerical_features].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ---- 7. Categorical Feature Distribution ----
categorical_cols = df.select_dtypes(include="object").columns.tolist()
categorical_cols = ["SeniorCitizen"] + [col for col in categorical_cols if col != "customerID"]

for col in categorical_cols:
    plt.figure(figsize=(6, 3))
    sns.countplot(data=df, x=col, palette="pastel")
    plt.title(f"Count Plot of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ---- 8. Churn Distribution ----
sns.countplot(data=df, x="Churn", palette="Set2")
plt.title("Churn Distribution")
plt.show()

print("\nChurn Rate (%):")
print(df["Churn"].value_counts(normalize=True) * 100)
