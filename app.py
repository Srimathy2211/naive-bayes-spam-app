from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("nb_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])
    prediction = model.predict(final_features)
    label = "Likely to Respond" if prediction[0] == 1 else "Unlikely to Respond"
    return render_template("index.html", prediction_text=f"Prediction: {label}")

if __name__ == "__main__":
    from os import environ
    app.run(host="0.0.0.0",port=int(environ.get("PORT",5000)))
