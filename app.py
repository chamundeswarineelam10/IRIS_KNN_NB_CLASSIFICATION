from flask import Flask, render_template, request
import numpy as np
import pickle
import json

app = Flask(__name__)

# Load models
knn_model = pickle.load(open("KNN_iris_model.pkl", "rb"))
nb_model = pickle.load(open("NB_iris_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values
        sl = float(request.form["sl"])
        sw = float(request.form["sw"])
        pl = float(request.form["pl"])
        pw = float(request.form["pw"])
        model_type = request.form["model"]

        features = np.array([[sl, sw, pl, pw]])

        # Select model
        if model_type == "knn":
            model = knn_model
            train_file = "KNN_train_result.json"
            test_file = "KNN_test_result.json"
        else:
            model = nb_model
            train_file = "NB_train_result.json"
            test_file = "NB_test_result.json"

        # Prediction
        prediction = model.predict(features)[0]

        species = {
            0: "Iris Setosa",
            1: "Iris Versicolor",
            2: "Iris Virginica"
        }

        result = species.get(prediction, "Unknown")

        # Load JSON files
        with open(train_file, "r") as f:
            train_data = json.load(f)

        with open(test_file, "r") as f:
            test_data = json.load(f)

        return render_template(
            "index.html",
            prediction_text=result,
            train_accuracy=train_data["accuracy"],
            test_accuracy=test_data["accuracy"],
            train_cm=train_data["confusion_matrix"],
            test_cm=test_data["confusion_matrix"],
            train_report=train_data["classification_report"],
            test_report=test_data["classification_report"]
        )

    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
