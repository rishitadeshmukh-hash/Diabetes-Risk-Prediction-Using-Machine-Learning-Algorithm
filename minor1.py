import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import gradio as gr

# ==========================================
# 1. Data Collection and Analysis
# ==========================================

# Loading the diabetes dataset from your local directory
# Make sure 'diabetes.csv' is in the same folder as this script
diabetes_dataset = pd.read_csv('diabetes.csv')

# Data exploration summaries
print("Dataset Head:\n", diabetes_dataset.head())
print("\nDataset Shape:", diabetes_dataset.shape)
print("\nStatistical Measures:\n", diabetes_dataset.describe())
print("\nOutcome Counts:\n", diabetes_dataset['Outcome'].value_counts())
print("\nGroupby Means:\n", diabetes_dataset.groupby('Outcome').mean())

# Separating the data and labels
X = diabetes_dataset.drop(columns='Outcome')
Y = diabetes_dataset['Outcome']

# ==========================================
# 2. Data Standardization & Train-Test Split
# ==========================================

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data
Y = diabetes_dataset['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print("\nData Split Shapes (Total, Train, Test):", X.shape, X_train.shape, X_test.shape)

# ==========================================
# 3. Training the Model
# ==========================================

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('\nAccuracy score of the training data : ', training_data_accuracy)

# Accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

# ==========================================
# 4. Making a Static Prediction System
# ==========================================

input_data = (5, 166, 72, 19, 175, 25.08, 0.587, 51)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)

prediction = classifier.predict(std_data)
if prediction[0] == 0:
    print('\nStatic test check: The person is not diabetic')
else:
    print('\nStatic test check: The person is diabetic')

# ==========================================
# 5. Gradio Web Interface
# ==========================================

def predict_diabetes(age, bmi, glucose, bp, insulin):
    # This hardcoded logic is kept from your original code.
    # Note: To use your actual trained SVM model instead, you would need to collect 
    # all 8 features matching your dataset columns and pass them through scaler.transform().
    risk_score = (glucose * 0.4) + (bmi * 0.3) + (age * 0.2)
    if risk_score > 70:
        return "High Risk of Diabetes. Please consult a doctor."
    elif risk_score > 40:
        return "Moderate Risk. Consider lifestyle improvements."
    else:
        return "Low Risk. Maintain your healthy habits!"

interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Slider(minimum=1, maximum=100, value=30, label="Age (years)"),
        gr.Slider(minimum=10, maximum=50, value=24, label="BMI (Body Mass Index)"),
        gr.Number(label="Glucose Level (mg/dL)", value=100),
        gr.Number(label="Blood Pressure (mm hg)", value=80),
        gr.Number(label="Insulin Level (uU/mL)", value=80)
    ],
    outputs=gr.Textbox(label="Risk Assessment Result"),
    title="Diabetes Risk Predictor",
    description="Enter patient health metrics to analyze diabetes risk factors instantly.",
    theme="soft"
)

# Launching locally. Setting share=False since you are running locally on your own machine.
if __name__ == "__main__":
    interface.launch(share=False)
    dd