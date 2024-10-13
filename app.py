from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('carbon_intensity_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        carbon_intensity = float(request.form['carbon_intensity'])
        energy_consumption = float(request.form['energy_consumption'])

        # Create a DataFrame for prediction
        new_data = pd.DataFrame({
            'carbon_intensity': [carbon_intensity],
            'energy_consumption': [energy_consumption]
        })

        # Predict carbon intensity and calculate carbon footprint
        predicted_intensity = model.predict(new_data)
        carbon_footprint = energy_consumption * predicted_intensity[0]

        return render_template('result.html', predicted_intensity=predicted_intensity[0], carbon_footprint=carbon_footprint)

    except Exception as e:
        return f"Error in prediction: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
