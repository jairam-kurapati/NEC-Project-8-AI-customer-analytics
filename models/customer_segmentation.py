import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("dataset/customers.csv")

features = df[["AnnualIncome", "SpendingScore"]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(scaled_features)

joblib.dump(kmeans, "saved_models/kmeans.pkl")
joblib.dump(scaler, "saved_models/scaler.pkl")

print("Model trained successfully!")
print(df[["CustomerID", "Cluster"]].head())