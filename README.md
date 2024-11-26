# MLZOOCAMP Midterm Project - Loan Eligibility Prediction
This project was developed as part of the MLZOOCAMP midterm evaluation. It is a machine learning application that predicts loan eligibility based on user-provided financial and demographic information. The project is deployed on Render and uses Flask for the backend, alongside a trained machine learning model.

## Features
User Interface: A web-based interface for submitting loan applications.

Real-time Predictions: Determines whether a loan request is likely to be approved ("Paid") or not ("Not Paid").

Scalable Backend: Built using Flask and deployed on Render for easy accessibility.

Validation: Inputs are rigorously validated to ensure data quality and consistency.

Machine Learning Model: A pre-trained model processes user data and predicts outcomes based on historical trends.

## Project Details
Input Features

The application accepts the following details:

- Gender (M, F)

- Type of Residence (Rented Apartment, Own House, Parents Apartment)

- Educational Attainment (BSc, HND & Other Equivalent, Diploma, Diploma/School Cert, MSc and Above, Others, School Cert)
- Employment Status (Employed, Self Employed, Others)
- Sector of Employment (Others, Wholesale and Retail Trade, Manufacturing and Construction, Agriculture, Education, Information Technology, Transportation & Logistics)
- Requested Loan Amount (numeric, between 100 and 2,500,000)
- Loan Purpose (Business, Medical, Others, Personal)
- Loan Request Day (Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday)
- Age (numeric, between 18 and 67)
- Selfie ID Check Status (Successful, Others)
- Number of Loans (numeric)
- Phone Numbers Available (numeric, between 0 and 5)
- Mobile Operating System (android, ios, Others)
- Income Range:
    - Medium-Low(50000-99999)
    - Low(0-49999)
    - Medium(10000-199999)
    - Very High(400000+)
    - High(200000-399999)
- Output
The application predicts whether a loan will be Paid or Not Paid based on the input data.

### Technology Stack
- Frontend: HTML templates for user input forms.
- Backend: Flask application to handle requests and predictions.
- Machine Learning: Pre-trained Logistic Regression model.
- Feature scaling and categorical encoding for accurate predictions.
- Deployment: Render platform for web hosting.
- Dependencies:
    - Flask
    - Joblib
    - Pandas
    - Numpy
    - Gunicorn

## Deployment
The project is deployed on Render and can be accessed [here](https://ml-zoocamp-midterm-project.onrender.com)

![image](https://github.com/user-attachments/assets/3a52f247-45eb-4722-83b2-0610b979f6fa)
