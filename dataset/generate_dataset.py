import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

data = {
    "CustomerID": range(1, n + 1),
    "Age": np.random.randint(18, 65, n),
    "AnnualIncome": np.random.randint(20000, 150000, n),
    "SpendingScore": np.random.randint(1, 100, n),
    "PurchaseFrequency": np.random.randint(1, 50, n),
    "TotalSpent": np.random.randint(1000, 100000, n),
}

df = pd.DataFrame(data)

df["Purchase"] = (
    (df["SpendingScore"] > 50) &
    (df["PurchaseFrequency"] > 20)
).astype(int)

df["Churn"] = (
    (df["SpendingScore"] < 30) &
    (df["PurchaseFrequency"] < 10)
).astype(int)

df.to_csv("dataset/customers.csv", index=False)

print("Dataset generated successfully!")
print(df.head())