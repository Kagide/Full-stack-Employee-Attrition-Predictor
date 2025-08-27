# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def generate_sample_data():
    """Generate sample employee data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.randint(22, 65, n_samples),
        'Gender': np.random.choice(['Male', 'Female', 'Non-binary'], n_samples, p=[0.6, 0.35, 0.05]),
        'Marital Status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples),
        'Department': np.random.choice(['Sales', 'Finance', 'HR', 'Operations', 'Marketing', 'IT'], n_samples),
        'Job Role': np.random.choice(['Developer', 'Manager', 'Analyst', 'Sales Rep', 'Team Lead'], n_samples),
        'Salary': np.random.randint(30000, 150000, n_samples),
        'Performance Rating': np.random.randint(1, 6, n_samples),
        'Years at Company': np.random.randint(0, 25, n_samples),
        'Promotion Count': np.random.randint(0, 6, n_samples),
        'Work-Life Balance Rating': np.random.randint(1, 6, n_samples),
        'Job Satisfaction': np.random.randint(1, 6, n_samples),
        'Training Hours': np.random.randint(0, 101, n_samples),
        'Commute Distance': np.random.uniform(1, 50, n_samples).round(1),
        'Absenteeism Rate': np.random.uniform(0, 30, n_samples).round(1),
        'Company Culture Fit': np.random.randint(1, 6, n_samples),
        'Team Dynamics': np.random.randint(1, 6, n_samples),
        'Company Loyalty': np.random.randint(1, 6, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable based on features (simulating attrition risk)
    df['Attrition'] = 0  # Default to no attrition
    
    # Higher risk conditions
    high_risk_mask = (
        (df['Job Satisfaction'] <= 2) |
        (df['Work-Life Balance Rating'] <= 2) |
        (df['Salary'] < 50000) |
        (df['Promotion Count'] == 0) & (df['Years at Company'] > 3) |
        (df['Company Culture Fit'] <= 2)
    )
    
    df.loc[high_risk_mask, 'Attrition'] = 1
    
    # Add some randomness
    random_mask = np.random.random(n_samples) < 0.1
    df.loc[random_mask, 'Attrition'] = 1 - df.loc[random_mask, 'Attrition']
    
    return df

def train_employee_attrition_model():
    """Train and save the employee attrition model"""
    print("Generating sample data...")
    df = generate_sample_data()
    
    print("Preprocessing data...")
    # Separate features and target
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # Encode categorical variables
    categorical_columns = ['Gender', 'Marital Status', 'Department', 'Job Role']
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_columns = [col for col in X.columns if col not in categorical_columns]
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
    
    print("Training model...")
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model artifacts
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': list(X.columns),
        'accuracy': accuracy
    }
    
    joblib.dump(model_artifacts, 'employee_attrition_model.joblib')
    print("Model saved as 'employee_attrition_model.joblib'")
    
    return model_artifacts

if __name__ == "__main__":
    train_employee_attrition_model()