import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.Education.replace({'Bachelors': 1, 'Masters': 2, 'PHD': 3}, inplace=True)
    df.Gender.replace({'Male': 0, 'Female': 1}, inplace=True)
    df.EverBenched.replace({'Yes': 1, 'No': 0}, inplace=True)
    return df

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print('Accuracy:', accuracy)
    print('Confusion Matrix:\n', cm)

    return y_pred

def predict_single_instance(model, input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    
    return prediction

def main():
    # Load and prepare data
    df = load_and_prepare_data('datasets/employee.csv')
    X = df.drop(columns=['City', 'LeaveOrNot'], axis=1)
    y = df.LeaveOrNot

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the logistic regression model
    logistic_model = train_logistic_regression(X_train, y_train)

    # Evaluate the model
    evaluate_model(logistic_model, X_test, y_test)

    # Predict a single instance
    input_data = [1, 2018, 3, 34, 0, 0, 0]  # Example input data
    prediction = predict_single_instance(logistic_model, input_data)

    print('Predicted Value:', prediction[0])
    if prediction[0] == 0:
        print('The person will not leave the company')
    else:
        print('The person will leave the company')

if __name__ == "__main__":
    main()
