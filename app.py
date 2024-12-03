import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the machine learning model
model_path = 'diabetes_model.pkl'

# Load the model with error handling
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    raise Exception(f"Model file not found at {model_path}. Ensure the file exists.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features in the correct order
        input_data = [
            float(request.form['high_bp']),
            float(request.form['high_chol']),
            float(request.form['bmi']),
            float(request.form['smoker']),
            float(request.form['stroke']),
            float(request.form['heart_disease']),
            float(request.form['phys_activity']),
            float(request.form['gen_health']),
            float(request.form['ment_health']),
            float(request.form['phys_health']),
            float(request.form['diff_walk']),
            float(request.form['sex']),
            float(request.form['age']),
            float(request.form['education']),
            float(request.form['income'])
        ]

        # Make prediction
        prediction = model.predict([input_data])
        result = 'Positive' if prediction[0] == 1 else 'Negative'

        # Render results with appropriate feedback
        if result == 'Positive':
            return render_template(
                'results.html',
                prediction=result,
                advice="You may have diabetes. Please consult a healthcare provider.",
                links=[
                    {"name": "American Diabetes Association", "url": "https://diabetes.org/about-diabetes/diagnosis"},
                    {"name": "Mayo Clinic", "url": "https://www.mayoclinic.org/diseases-conditions/diabetes/diagnosis-treatment/drc-20371451"},
                    {"name": "OptiMind Health", "url": "https://www.optimindhealth.com/"}
                ]
            )
        else:
            return render_template(
                'results.html',
                prediction=result,
                advice="Congratulations! You do not have diabetes.",
                show_fireworks=True
            )
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
