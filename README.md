Employee Attrition Prediction System
A machine learning web application that predicts employee turnover risk with 87% accuracy. This tool helps HR teams proactively identify at-risk employees and reduce turnover.

✨ Features
📊 Single Employee Prediction: Real-time risk assessment with interactive form

📁 Batch Processing: Handle thousands of employees with progress tracking

📈 Interactive Dashboard: Beautiful visualizations and analytics

👥 HR Analytics: Department-wise insights and trends

⚡ Live Demo: Fully deployed and accessible online

🛠️ Technical Stack
Python: Scikit-learn, Pandas, NumPy

Web Framework: Streamlit

Visualization: Plotly

ML Model: Random Forest Classifier

Deployment: Streamlit Community Cloud

🚀 Quick Start
Prerequisites
Python 3.8+

pip package manager

Installation
Clone the repository

bash
git clone https://github.com/YOUR_USERNAME/employee-attrition-predictor.git
cd employee-attrition-predictor
Install dependencies:

bash
pip install -r requirements.txt
Train the model (if not included):

bash
python train_model.py
Run the application locally:

bash
streamlit run app.py
🌐 Live Deployment
Experience the full application live:
https://your-app-name.streamlit.app/

📁 Project Structure
text
employee-attrition-predictor/
├── app.py                 # Main Streamlit application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── employee_attrition_model.joblib  # Trained model
├── README.md              # This file
└── assets/               # Images and demo files
    └── demo.gif          # Application demo
📊 Model Performance
Accuracy: 87%

Algorithm: Random Forest Classifier

Features: 17 employee attributes including:

Job Satisfaction

Work-Life Balance

Salary and Compensation

Years at Company

Performance Metrics

🎯 Usage Examples
Single Employee Prediction
Fill in employee details in the interactive form

Get instant risk assessment with probability percentage

View detailed visual explanations

Batch Processing
Upload CSV file with employee data

Monitor real-time processing with progress bar

Download comprehensive results with risk analysis

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the project

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Ellevo Pathways for the machine learning internship opportunity

Coursera for the Supervised Machine Learning course

Streamlit for the amazing web framework

Scikit-learn team for the machine learning library

