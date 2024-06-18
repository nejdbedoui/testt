import pandas as pd
import pickle
from flask import Flask, request, jsonify

# Load the trained model from a file
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Define a route to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.json

    # Convert JSON data to a DataFrame
    user_df = pd.DataFrame([data])

    # Make prediction using the loaded model
    predicted_category = best_model.predict(user_df)[0]

    # Return the predicted category as a JSON response
    return jsonify({'predicted_category': predicted_category})

if __name__ == '__main__':
    app.run(debug=True)
