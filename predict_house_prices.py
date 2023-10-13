# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your dataset (you'd typically load real-world data)
data = {
    'Bedrooms': [2, 3, 4, 3, 2, 4, 2, 3, 4, 3],
    'SquareFeet': [1000, 1500, 2000, 1200, 800, 1800, 950, 1400, 2100, 1300],
    'Neighborhood': ['A', 'B', 'C', 'B', 'A', 'C', 'A', 'B', 'C', 'B'],
    'Price': [200000, 300000, 400000, 250000, 180000, 380000, 190000, 290000, 420000, 270000]
}

df = pd.DataFrame(data)

# Convert categorical variable (Neighborhood) into numerical
df = pd.get_dummies(df, columns=['Neighborhood'], prefix=['Neighborhood'])

# Split data into features (X) and target (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# You can now use this model to predict house prices in real-world scenarios
