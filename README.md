 **Loan Approval Prediction System**

This project uses machine learning to predict whether a loan application should be approved or rejected based on an applicant's financial and demographic profile. It is built using the popular **Kaggle Loan Prediction dataset**, which contains real-world features commonly used in banking and financial risk assessment.

The system employs a **tuned Random Forest classifier** with advanced feature engineering to achieve high predictive accuracy and robustness.

---

## Features

- **Advanced Feature Engineering**: Creates composite features like `TotalIncome`, `LoanAmountToIncomeRatio`, and `EMI` for better model performance.
- **Robust Preprocessing Pipeline**: Handles missing values, scales numerical data, and encodes categorical variables automatically.
- **Hyperparameter Tuning**: Uses `GridSearchCV` to optimize the model for maximum ROC-AUC score.
- **Business-Ready Risk Tiers**: The prediction function doesn't just output "Yes/No"; it provides a probability and categorizes applicants into risk tiers (e.g., `LOW_RISK_AUTO_APPROVE`, `VERY_HIGH_RISK_REJECT`).
- **Production-Ready Model**: The trained model is saved as `model.pkl` for easy deployment.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SHEVISANTOS/loan.git
   cd loan
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```



###  Usage

#### 1. Explore the Project
Open the `notebook.ipynb` file in Jupyter Lab or VS Code to see the full end-to-end workflow:
- Data cleaning and exploration
- Feature engineering
- Model training, tuning, and evaluation
- Final deployment logic

#### 2. Use the Trained Model
The pre-trained model (`model.pkl`) can be loaded to make predictions on new data.

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('model.pkl')

# Prepare a single applicant's data as a dictionary
applicant = {
    'Gender': 'Male', 'Married': 'Yes', 'Dependents': 2.0, 'Education': 'Graduate',
    'Self_Employed': 'No', 'ApplicantIncome': 12000, 'CoapplicantIncome': 5000,
    'LoanAmount': 200, 'Loan_Amount_Term': 360.0, 'Credit_History': 1.0,
    'Property_Area': 'Urban'
}

# The model's pipeline will automatically engineer the required features
df = pd.DataFrame([applicant])
prediction = model.predict(df)
probability = model.predict_proba(df)[:, 1]

print(f"Approved: {bool(prediction[0])}, Probability: {probability[0]:.2%}")
```

> **Note**: The model expects the **original 11 features** from the dataset. The engineered features (`TotalIncome`, etc.) are created internally by the preprocessing pipeline.

---

### Project Structure
```
loan/
├── loan.csv              # The original Kaggle Loan Prediction dataset
├── model.pkl             # The final, tuned Random Forest model (saved with joblib)
├── notebook.ipynb        # Complete Jupyter notebook with all code and analysis
├── requirements.txt      # Python dependencies
└── venv/                 # Virtual environment (should be in .gitignore)
```

---

### 📊 Model Performance
After hyperparameter tuning, the model achieves:
- **Accuracy**: ~81.3%
- **ROC-AUC**: ~81.8%

The most important features for prediction are:
1. `CreditScore_1` (a strong indicator of creditworthiness)
2. `LoanAmountToIncomeRatio` (measures financial burden)
3. `TotalIncome` (overall financial capacity)

These results align with real-world lending practices where credit history and debt-to-income ratio are critical factors [[1]].

---



---

### ✉️ Contact
Built by **Shevi Maisha Jeremiah**  
📧 shevijeremiah@gmail.com  
🔗 [GitHub Profile](https://github.com/SHEVISANTOS)