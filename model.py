import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("house_data.csv")

# Input features
X = data[['area', 'bedrooms']]

# Output target
y = data['price']

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Save trained model
pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained successfully!")