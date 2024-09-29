from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))
scaler=pickle.load(open('scaler.pkl', 'rb'))
# Define the mappings
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
tech_support_map = {'No': 0, 'Yes': 1, 'No internet service': 2}
online_security_map = {'No': 0, 'Yes': 1, 'No internet service': 2}
paperless_billing_map = {'Yes': 0, 'No': 1}
senior_citizen_map = {'No': 0, 'Yes': 1}

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form and map categorical values to numerical ones
          contract = contract_map.get(request.form['Contract'])
          tenure = int(request.form['tenure'])
          tech_support = tech_support_map.get(request.form['TechSupport'])
          online_security = online_security_map.get(request.form['OnlineSecurity'])
          total_charges = float(request.form['TotalCharges'])
          paperless_billing = paperless_billing_map.get(request.form['PaperlessBilling'])
          senior_citizen = senior_citizen_map.get(request.form['SeniorCitizen'])
          monthly_charges = float(request.form['MonthlyCharges'])
          numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
          df = pd.DataFrame({
              'tenure': [tenure],
              'MonthlyCharges': [monthly_charges],
              'TotalCharges': [total_charges]
          })
          df[numerical_cols]=scaler.transform(df[numerical_cols])
          tenure=df['tenure'][0]
          monthly_charges=df['MonthlyCharges'][0]
          total_charges=df['TotalCharges'][0]
          # Collect all the variables into a list
          input_list = [
              contract, 
              tenure, 
              tech_support, 
              online_security, 
              total_charges, 
              paperless_billing, 
              senior_citizen, 
              monthly_charges
          ]
          data = {
            'Contract':contract,
            'tenure': tenure,
            'TechSupport': tech_support,
            'OnlineSecurity': online_security,
            'TotalCharges': total_charges,
            'PaperlessBilling': paperless_billing,
            'SeniorCitizen': senior_citizen,
            'MonthlyCharges': monthly_charges
        }

        # Check for missing or incorrect mappings
          for key, value in data.items():
            if value is None:
                return jsonify({'error': f"Invalid input for {key}: {request.form[key]}"})

        # Convert the data into a pandas DataFrame to match the model's expected input format
          input_data = pd.DataFrame([data])

        # Perform prediction using the trained model
          prediction = model.predict(input_data)

        # Interpret the result
          output = 'Churn' if prediction[0] == 1 else 'No Churn'
        
          return render_template('index.html', prediction_text=f'Customer will: {output}')
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
