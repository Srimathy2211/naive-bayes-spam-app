import pandas as pd
from sklearn.naive_bayes import GaussianNB
import pickle

# Simulated dataset: Predict email response likelihood
data = pd.DataFrame({
    "length": [50, 300, 150, 20, 500, 100, 400, 30],
    "urgency": [3, 1, 2, 5, 1, 4, 1, 5],  # 1=low urgency, 5=very urgent
    "attachments": [0, 2, 1, 0, 3, 0, 1, 0],
    "hour_sent": [9, 14, 8, 23, 15, 10, 16, 22],  # 0-23 hour
    "response": [1, 0, 1, 0, 0, 1, 0, 0]
})

X = data.drop("response", axis=1)
y = data["response"]

model = GaussianNB()
model.fit(X, y)

with open("nb_model.pkl", "wb") as f:
    pickle.dump(model, f)
