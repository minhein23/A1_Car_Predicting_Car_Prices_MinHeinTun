from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model
model = joblib.load('random_forest_model.pkl')  # Load the trained model

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Render the homepage with the form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        year = int(request.form['year'])
        engine = float(request.form['engine'])
        max_power = float(request.form['max_power'])

        # Prepare the features for prediction
        features = np.array([[year, engine, max_power]])

        # Predict the car price using the model
        prediction = model.predict(features)

        # Reverse the log transformation (if applied)
        predicted_price = np.exp(prediction[0])  # Reverse log if you log-transformed the target

        return render_template('index.html', predicted_price=f"Predicted Car Price: {predicted_price:,.2f}")
    
    except Exception as e:
        return render_template('index.html', error_message=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
