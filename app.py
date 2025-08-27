# app.py - ENHANCED VERSION
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import io

# Configure page
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .low-risk {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
    }
    .progress-info {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .high-risk-card {
        border-left: 4px solid #dc3545;
    }
    .low-risk-card {
        border-left: 4px solid #28a745;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return joblib.load('employee_attrition_model.joblib')
    except FileNotFoundError:
        st.error("Model file not found! Please run train_model.py first.")
        return None

def predict_single(model_artifacts, employee_data):
    if model_artifacts.get('model') is None:
        # Demo mode - simple logic based on job satisfaction
        job_satisfaction = employee_data.get('Job Satisfaction', 3)
        pred = 1 if job_satisfaction <= 2 else 0
        prob = [0.8, 0.2] if pred == 1 else [0.2, 0.8]
        return pred, prob
    
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    label_encoders = model_artifacts['label_encoders']
    feature_names = model_artifacts['feature_names']
    
    # Create DataFrame
    df = pd.DataFrame([employee_data])
    
    # Encode categorical variables
    categorical_columns = ['Gender', 'Marital Status', 'Department', 'Job Role']
    for col in categorical_columns:
        if col in label_encoders and col in df.columns:
            try:
                df[col] = label_encoders[col].transform(df[col])
            except ValueError:
                df[col] = 0
    
    # Ensure all features present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    df = df[feature_names]
    
    # Scale numerical features
    numerical_columns = [col for col in df.columns if col not in categorical_columns]
    df[numerical_columns] = scaler.transform(df[numerical_columns])
    
    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]
    
    return prediction, probability

# Enhanced Data Validation
def validate_employee_data(df):
    """Validate uploaded CSV has required columns"""
    required_columns = ['Age', 'Job Satisfaction']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Validate data types and ranges
    if 'Age' in df.columns:
        invalid_ages = df[(df['Age'] < 18) | (df['Age'] > 70)]
        if len(invalid_ages) > 0:
            return False, f"Found {len(invalid_ages)} employees with age outside valid range (18-70)"
    
    return True, "Data validation passed"

def single_prediction_page(model_artifacts):
    st.header("Single Employee Attrition Prediction")
    
    with st.form("employee_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Basic Info")
            age = st.slider("Age", 18, 70, 35)
            gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
            
        with col2:
            st.subheader("Work Info")
            department = st.selectbox("Department", ["Sales", "Finance", "HR", "Operations", "Marketing", "IT"])
            job_role = st.selectbox("Job Role", ["Developer", "Manager", "Analyst", "Sales Rep", "Team Lead"])
            years = st.slider("Years at Company", 0, 40, 5)
            salary = st.number_input("Annual Salary ($)", 30000, 200000, 70000)
            
        with col3:
            st.subheader("Performance")
            performance = st.slider("Performance Rating", 1, 5, 3)
            satisfaction = st.slider("Job Satisfaction", 1, 5, 3)
            work_life = st.slider("Work-Life Balance", 1, 5, 3)
            culture_fit = st.slider("Company Culture Fit", 1, 5, 3)
        
        col4, col5 = st.columns(2)
        with col4:
            promotions = st.number_input("Promotions", 0, 10, 1)
            training = st.slider("Training Hours/Year", 0, 100, 25)
            team_dynamics = st.slider("Team Dynamics", 1, 5, 3)
            
        with col5:
            commute = st.slider("Commute Distance", 0.0, 100.0, 15.0)
            absenteeism = st.slider("Absenteeism Rate %", 0.0, 100.0, 10.0)
            loyalty = st.slider("Company Loyalty", 1, 5, 3)
        
        submitted = st.form_submit_button("Predict Attrition Risk")
        
        if submitted:
            employee_data = {
                'Age': age, 'Gender': gender, 'Marital Status': marital_status,
                'Department': department, 'Job Role': job_role, 'Salary': salary,
                'Performance Rating': performance, 'Years at Company': years,
                'Promotion Count': promotions, 'Work-Life Balance Rating': work_life,
                'Job Satisfaction': satisfaction, 'Training Hours': training,
                'Commute Distance': commute, 'Absenteeism Rate': absenteeism,
                'Company Culture Fit': culture_fit, 'Team Dynamics': team_dynamics,
                'Company Loyalty': loyalty
            }
            
            try:
                prediction, prob = predict_single(model_artifacts, employee_data)
                attrition_prob = prob[1] * 100
                
                st.subheader("Prediction Results")
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box high-risk">
                        <h3>HIGH ATTRITION RISK</h3>
                        <p><strong>Probability:</strong> {attrition_prob:.1f}%</p>
                        <p><strong>Action:</strong> Immediate attention required!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box low-risk">
                        <h3>LOW ATTRITION RISK</h3>
                        <p><strong>Probability:</strong> {attrition_prob:.1f}%</p>
                        <p><strong>Status:</strong> Employee likely to stay</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = attrition_prob,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Attrition Risk %"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "red" if prediction == 1 else "green"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {e}")

def batch_prediction_page(model_artifacts):
    st.header("Batch Employee Prediction")
    
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            # Validate the data
            is_valid, message = validate_employee_data(df)
            if not is_valid:
                st.error(f"Data validation failed: {message}")
                return
                
            st.write(f"Loaded {len(df):,} employees")
            st.dataframe(df.head())
            
            # Show file size warning with accurate estimation
            if len(df) > 500:
                # Realistic estimation: ~20-100ms per employee
                est_seconds = len(df) * 0.05  # 50ms per employee
                est_minutes = est_seconds / 60
                
                if est_minutes > 2:
                    st.warning(f"Large file detected ({len(df):,} employees). Processing may take approximately {est_minutes:.1f} minutes.")
                else:
                    st.info(f"Large file detected ({len(df):,} employees). Processing may take approximately {est_seconds:.1f} seconds.")
            
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            return
        
        if st.button("Generate Predictions"):
            # Create containers for progress indicators
            progress_container = st.container()
            status_container = st.container()
            
            with progress_container:
                st.markdown("### Processing Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_estimate_text = st.empty()
                speed_text = st.empty()
                errors_text = st.empty()
                
            start_time = time.time()
            predictions = []
            probabilities = []
            errors = 0
            error_messages = []
            
            # Process each row individually to show progress
            total_rows = len(df)
            
            for i, (idx, row) in enumerate(df.iterrows()):
                # Update progress
                progress = (i + 1) / total_rows
                progress_bar.progress(progress)
                
                # Calculate time estimates
                elapsed_time = time.time() - start_time
                if i > 0:
                    time_per_row = elapsed_time / (i + 1)
                    remaining_time = time_per_row * (total_rows - i - 1)
                    processed_per_second = (i + 1) / elapsed_time if elapsed_time > 0 else 0
                    
                    status_text.markdown(f"**Processing row {i+1} of {total_rows}**")
                    
                    # Format time nicely
                    if remaining_time > 120:
                        time_display = f"{remaining_time/60:.1f} minutes"
                    else:
                        time_display = f"{remaining_time:.1f} seconds"
                    
                    time_estimate_text.markdown(f"<div class='progress-info'>⏱️ **Est. remaining:** {time_display}</div>", unsafe_allow_html=True)
                    speed_text.markdown(f"<div class='progress-info'>🚀 **Speed:** {processed_per_second:.1f} rows/second</div>", unsafe_allow_html=True)
                    errors_text.markdown(f"<div class='progress-info'>❌ **Errors:** {errors}</div>", unsafe_allow_html=True)
                
                # Process the row
                try:
                    pred, prob = predict_single(model_artifacts, row.to_dict())
                    predictions.append("HIGH RISK" if pred == 1 else "LOW RISK")
                    probabilities.append(f"{prob[1]*100:.1f}%")
                except Exception as e:
                    predictions.append("ERROR")
                    probabilities.append("N/A")
                    errors += 1
                    error_messages.append(f"Row {i+1}: {str(e)}")
                    # Only show occasional errors to avoid flooding the UI
                    if errors <= 5 or errors % 10 == 0:
                        st.error(f"Error processing row {i+1}: {str(e)}")
                
                # Add a small delay to make progress visible (only in demo mode)
                if model_artifacts.get('model') is None:
                    time.sleep(0.01)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            time_estimate_text.empty()
            speed_text.empty()
            errors_text.empty()
            
            # COMPLETION MESSAGE - Shows total time
            total_time = time.time() - start_time
            st.success(f"✅ **Completed in {total_time:.1f} seconds!** ({errors} errors)")
            
            # Add results to dataframe
            df['Attrition_Prediction'] = predictions
            df['Risk_Probability'] = probabilities
            
            # PREDICTION SUMMARY with metrics
            st.subheader("Prediction Summary")
            
            # Calculate stats
            high_risk = sum(1 for p in predictions if p == "HIGH RISK")
            low_risk = sum(1 for p in predictions if p == "LOW RISK")
            total_valid = high_risk + low_risk
            
            # Enhanced metrics display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div>Total Processed</div>
                    <div class="stat-number">{total_valid}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                risk_percent = f"{high_risk/total_valid*100:.1f}%" if total_valid > 0 else "0%"
                st.markdown(f"""
                <div class="metric-card high-risk-card">
                    <div>High Risk</div>
                    <div class="stat-number" style="color: #dc3545;">{high_risk}</div>
                    <div>{risk_percent}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                risk_percent = f"{low_risk/total_valid*100:.1f}%" if total_valid > 0 else "0%"
                st.markdown(f"""
                <div class="metric-card low-risk-card">
                    <div>Low Risk</div>
                    <div class="stat-number" style="color: #28a745;">{low_risk}</div>
                    <div>{risk_percent}</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div>Errors</div>
                    <div class="stat-number">{errors}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # PIE CHART - Visual risk distribution
            if total_valid > 0:
                chart_data = pd.DataFrame({
                    'Risk Level': ['High Risk', 'Low Risk'],
                    'Count': [high_risk, low_risk]
                })
                
                fig = px.pie(
                    chart_data,
                    values='Count',
                    names='Risk Level',
                    title="Risk Distribution",
                    color_discrete_map={'High Risk': 'red', 'Low Risk': 'green'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed results
            st.subheader("Detailed Results")
            
            # Show only key columns first
            display_columns = []
            if 'Employee ID' in df.columns:
                display_columns.append('Employee ID')
            elif 'EmployeeID' in df.columns:
                display_columns.append('EmployeeID')
            elif 'id' in df.columns:
                display_columns.append('id')
            else:
                display_columns.append(df.columns[0])  # Use first column as identifier
            
            display_columns.extend(['Attrition_Prediction', 'Risk_Probability'])
            
            # Add a few important columns if they exist
            important_cols = ['Job Satisfaction', 'Department', 'Years at Company', 'Salary']
            for col in important_cols:
                if col in df.columns and col not in display_columns:
                    display_columns.append(col)
            
            st.dataframe(df[display_columns].head(20))
            
            # Download button
            csv = df.to_csv(index=False)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"attrition_predictions_{timestamp}.csv"
            
            st.download_button(
                label="Download Complete Results",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
            
            # HIGH-RISK ALERTS - Warnings for risky employees
            if high_risk > 0:
                st.subheader("🚨 High Risk Employees - Take Action")
                
                # Create a high risk dataframe
                high_risk_df = df[df['Attrition_Prediction'] == 'HIGH RISK'].copy()
                
                # Try to get the best identifier column
                id_col = None
                for col in ['Employee ID', 'EmployeeID', 'id', 'Name', 'First Name']:
                    if col in high_risk_df.columns:
                        id_col = col
                        break
                
                if id_col is None:
                    id_col = high_risk_df.columns[0]
                
                # Show top 10 high-risk employees
                for idx, emp in high_risk_df.head(10).iterrows():
                    emp_id = emp.get(id_col, f'Row {idx+1}')
                    risk_prob = emp['Risk_Probability']
                    
                    # Get additional context if available
                    context = ""
                    if 'Job Satisfaction' in emp and pd.notna(emp['Job Satisfaction']):
                        context = f" (Job Satisfaction: {emp['Job Satisfaction']})"
                    elif 'Department' in emp and pd.notna(emp['Department']):
                        context = f" (Department: {emp['Department']})"
                    
                    st.warning(f"Employee {emp_id}: {risk_prob} risk{context} - Schedule retention meeting")
    else:
        st.info("Upload a CSV file to get started")
        
        # Show sample format
        st.subheader("Expected CSV Format")
        sample_df = pd.DataFrame({
            'Employee ID': ['E001', 'E002'],
            'Age': [30, 45],
            'Gender': ['Male', 'Female'],
            'Department': ['IT', 'Sales'],
            'Job Role': ['Developer', 'Manager'],
            'Salary': [75000, 85000],
            'Performance Rating': [4, 5],
            'Years at Company': [3, 8],
            'Job Satisfaction': [4, 2],
            'Work-Life Balance Rating': [3, 2],
            'Training Hours': [30, 25],
            'Commute Distance': [15.0, 12.5],
            'Absenteeism Rate': [5.0, 8.0],
            'Company Culture Fit': [4, 2]
        })
        st.dataframe(sample_df)
        
        # Download sample CSV
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="Download Sample CSV",
            data=csv,
            file_name="sample_employee_data.csv",
            mime="text/csv"
        )

def about_page():
    st.header("About This Model")
    
    st.markdown("""
    ## Model Performance
    - **Algorithm**: Random Forest Classifier
    - **Accuracy**: ~87%
    - **Features**: 17 employee attributes
    
    ## Key Factors
    1. **Job Satisfaction** - Most important predictor
    2. **Work-Life Balance** - Critical for retention
    3. **Company Culture Fit** - Affects long-term stay
    4. **Years at Company** - Experience matters
    5. **Salary** - Compensation competitiveness
    
    ## HR Recommendations
    - Monitor employees with satisfaction < 3/5
    - Focus on work-life balance programs
    - Regular compensation reviews
    - Career development opportunities
    - Manager training on culture building
    
    ## How to Use
    1. **Single Prediction**: Use the form for individual employees
    2. **Batch Processing**: Upload CSV files for multiple predictions
    3. **Take Action**: Focus on high-risk employees identified
    
    Built with Python, Scikit-learn, and Streamlit
    """)

def main():
    st.markdown('<h1 class="main-header">Employee Attrition Predictor</h1>', unsafe_allow_html=True)
    
    # Load model
    model_artifacts = load_model()
    if model_artifacts is None:
        st.warning("Using demo mode with sample data")
        # Create a mock model artifacts structure for demo purposes
        model_artifacts = {
            'model': None,
            'scaler': None,
            'label_encoders': None,
            'feature_names': None
        }
    
    # Enhanced Navigation with icons
    page = st.sidebar.radio("Navigation", 
                           ["📈 Single Prediction", "📊 Batch Prediction", "📋 About"],
                           index=0)
    
    if page == "📈 Single Prediction":
        single_prediction_page(model_artifacts)
    elif page == "📊 Batch Prediction":
        batch_prediction_page(model_artifacts)
    elif page == "📋 About":
        about_page()

if __name__ == "__main__":
    main()