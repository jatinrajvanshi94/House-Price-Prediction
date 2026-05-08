from flask import Flask, render_template, request
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    # Get values from form
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])

    # Prediction
    prediction = model.predict([[area, bedrooms]])

    output = round(prediction[0], 2)

    return render_template(
        'index.html',
        prediction_text=f'Predicted House Price: ₹ {output}'
    )

# Run server
if __name__ == "__main__":
    app.run(debug=True)