# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("loan_data.csv")

# Display first 5 rows
print("Dataset:")
print(data.head())

# Convert text data into numbers
label_encoder = LabelEncoder()

data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Married'] = label_encoder.fit_transform(data['Married'])
data['Education'] = label_encoder.fit_transform(data['Education'])
data['Property_Area'] = label_encoder.fit_transform(data['Property_Area'])
data['Loan_Status'] = label_encoder.fit_transform(data['Loan_Status'])

# Features (Input)
X = data.drop('Loan_Status', axis=1)

# Target (Output)
y = data['Loan_Status']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100)

# User input prediction
print("\nEnter Loan Details:")

gender = int(input("Gender (Male=1 Female=0): "))
married = int(input("Married (Yes=1 No=0): "))
education = int(input("Graduate=0 Not Graduate=1: "))
income = int(input("Applicant Income: "))
loan_amount = int(input("Loan Amount: "))
credit_history = int(input("Credit History (1=Good 0=Bad): "))
property_area = int(input("Property Area (Urban=2 Semiurban=1 Rural=0): "))

# Create input data
new_data = [[
    gender,
    married,
    education,
    income,
    loan_amount,
    credit_history,
    property_area
]]

# Prediction
prediction = model.predict(new_data)

# Output result
if prediction[0] == 1:
    print("\nLoan Approved")
else:
    print("\nLoan Not Approved")