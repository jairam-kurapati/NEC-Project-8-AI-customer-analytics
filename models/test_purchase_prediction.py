import joblib
import numpy as np

model = joblib.load("saved_models/purchase_model.pkl")

sample_customer = np.array([
    [25, 60000, 80, 25, 30000]
])

prediction = model.predict(sample_customer)

if prediction[0] == 1:
    print("Customer likely to purchase")
else:
    print("Customer unlikely to purchase")