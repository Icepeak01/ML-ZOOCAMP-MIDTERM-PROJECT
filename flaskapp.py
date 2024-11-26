from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import joblib
import pandas as pd

# Validation function
def validate_input(data):
    required_fields = ['gender', 'type_of_residence', 'educational_attainment', 'employment_status',
                       'sector_of_employment', 'requested_amount', 'purpose', 'loan_request_day',
                       'age', 'selfie_id_check', 'loans', 'phone_numbers', 'mobile_os', 'income_range']

    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing fields: {', '.join(missing_fields)}"

    # Validate categorical values
    valid_income_ranges = [
        "Medium-Low(50000-99999)",
        "Low(0-49999)",
        "Medium(10000-199999)",
        "Very High(400000+)",
        "High(200000-399999)"
    ]
    valid_genders = ["M", "F"]
    valid_residences = ["Rented Apartment", "Own House", "Parents Apartment"]
    valid_educations = [
        "BSc", "HND & Other Equivalent", "Diploma", "Diploma/School Cert",
        "MSc and Above", "Others", "School Cert"
    ]
    valid_employments = ["Employed", "Self Employed", "Others"]
    valid_purposes = ["Business", "Medical", "Others", "Personal"]
    valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    valid_os = ["android", "ios", "Others"]
    valid_sectors = [
        "Others", "Wholesale and Retail Trade", "Manufacturing and Construction",
        "Agriculture", "Education", "Information Technology", "Transportation & Logistics"
    ]
    valid_selfie_checks = ["Successful", "Others"]

    if data['gender'] not in valid_genders:
        return False, f"Invalid 'gender'. Must be one of {', '.join(valid_genders)}"
    if data['type_of_residence'] not in valid_residences:
        return False, f"Invalid 'type_of_residence'. Must be one of {', '.join(valid_residences)}"
    if data['educational_attainment'] not in valid_educations:
        return False, f"Invalid 'educational_attainment'. Must be one of {', '.join(valid_educations)}"
    if data['employment_status'] not in valid_employments:
        return False, f"Invalid 'employment_status'. Must be one of {', '.join(valid_employments)}"
    if data['purpose'] not in valid_purposes:
        return False, f"Invalid 'purpose'. Must be one of {', '.join(valid_purposes)}"
    if data['loan_request_day'] not in valid_days:
        return False, f"Invalid 'loan_request_day'. Must be one of {', '.join(valid_days)}"
    if data['mobile_os'] not in valid_os:
        return False, f"Invalid 'mobile_os'. Must be one of {', '.join(valid_os)}"
    if data['sector_of_employment'] not in valid_sectors:
        return False, f"Invalid 'sector_of_employment'. Must be one of {', '.join(valid_sectors)}"
    if data['selfie_id_check'] not in valid_selfie_checks:
        return False, f"Invalid 'selfie_id_check'. Must be one of {', '.join(valid_selfie_checks)}"
    if data['income_range'] not in valid_income_ranges:
        return False, f"Invalid 'income_range'. Must be one of {', '.join(valid_income_ranges)}"

    # Validate numeric fields
    numeric_fields = ['requested_amount', 'age', 'loans', 'phone_numbers']
    for field in numeric_fields:
        if not isinstance(data[field], (int, float)):
            return False, f"Invalid type for {field}, expected a number"

    # Range checks
    if not (100 <= data['requested_amount'] <= 2500000):
        return False, "Invalid 'requested_amount', must be between 100 and 2,500,000"
    if not (18 <= data['age'] <= 67):
        return False, "Invalid 'age', must be between 18 and 67"
    if not (0 <= data['phone_numbers'] <= 5):
        return False, "Invalid 'phone_numbers', must be between 0 and 5"

    return True, "Valid input"

# Prediction function
def return_prediction(model, scaler, col_name, data):
    # Extract and encode input data
    cat_features = ['gender', 'type_of_residence', 'educational_attainment', 'employment_status',
                    'sector_of_employment', 'purpose', 'loan_request_day', 'selfie_id_check',
                    'mobile_os', 'income_range']
    num_features = ['requested_amount', 'age', 'loans', 'phone_numbers']

    cat_df = pd.DataFrame([[data[feature] for feature in cat_features]], columns=cat_features)
    cat_encoded = pd.get_dummies(cat_df, drop_first=True)

    num_df = pd.DataFrame([[data[feature] for feature in num_features]], columns=num_features)

    df = pd.concat([num_df, cat_encoded], axis=1)
    df = df.reindex(columns=col_name, fill_value=0)

    # Scale numeric data and predict
    scaled_data = scaler.transform(df.values)
    prediction = model.predict(scaled_data)

    return 'Paid' if prediction == 1 else 'Not Paid'

# Flask app setup
lendsqr = Flask(__name__)

# Load pre-trained models and feature columns
lr_model = joblib.load('ldsqr_lr_model.pkl')
lr_scaler = joblib.load('ldsqr_scaler.pkl')
col_name = joblib.load('col_name.pkl')

@lendsqr.route("/")
def home():
    return render_template('home.html')

@lendsqr.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form
        content = {
            'gender': data['gender'],
            'type_of_residence': data['type_of_residence'],
            'educational_attainment': data['educational_attainment'],
            'employment_status': data['employment_status'],
            'sector_of_employment': data['sector_of_employment'],
            'requested_amount': float(data['requested_amount']),
            'purpose': data['purpose'],
            'loan_request_day': data['loan_request_day'],
            'age': int(data['age']),
            'selfie_id_check': data['selfie_id_check'],
            'loans': int(data['loans']),
            'phone_numbers': int(data['phone_numbers']),
            'mobile_os': data['mobile_os'],
            'income_range': data['income_range']
        }

        # Validate input
        is_valid, message = validate_input(content)
        if not is_valid:
            return render_template('prediction.html', prediction=message)

        # Generate prediction
        result = return_prediction(lr_model, lr_scaler, col_name, content)
        return render_template('prediction.html', prediction=f"Loan Status: {result}")
    return redirect(url_for('home'))

if __name__ == '__main__':
    lendsqr.run(debug=True)
