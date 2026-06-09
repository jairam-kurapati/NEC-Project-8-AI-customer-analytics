import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset/customers.csv")

# Features
X = df[[
    "Age",
    "AnnualIncome",
    "SpendingScore",
    "PurchaseFrequency",
    "TotalSpent"
]]

# Target
y = df["Purchase"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Purchase Prediction Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "saved_models/purchase_model.pkl")

print("Purchase Prediction Model Saved!")