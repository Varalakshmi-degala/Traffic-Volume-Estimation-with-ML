import numpy as np
import pickle
import joblib
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open(r'C:\Traffic Volume Estimation Project\Flask\models\model.pkl','rb'))
scale = pickle.load(open(r'C:\Traffic Volume Estimation Project\Flask\models\encoder.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")  # Home page

@app.route('/predict', methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        try:
            # Reading user inputs
            input_feature = [float(x) for x in request.form.values()]
            features_values = [np.array(input_feature)]

            # Feature names should match training order
            names = [
                'holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day',
                'hours', 'minutes', 'seconds'
            ]

            # Create DataFrame
            data = pd.DataFrame(features_values, columns=names)

            # Apply scaling
            data_scaled = scale.transform(data)
            data_scaled = pd.DataFrame(data_scaled, columns=names)

            # Predict
            prediction = model.predict(data_scaled)
            text = "Estimated Traffic Volume is: "

            return render_template("index.html", prediction_text=text + str(round(prediction[0], 2)))

        except Exception as e:
            return render_template("index.html", prediction_text="Error: " + str(e))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(port=5000, debug=True)
