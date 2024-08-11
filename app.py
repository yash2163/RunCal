from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Load the model and the scaler
model = pickle.load(open('best_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Ensure the scaler is loaded

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        weight = float(request.form.get('weight'))
        height = float(request.form.get('height'))
        duration = float(request.form.get('duration'))
        speed = float(request.form.get('speed'))

        # Prepare the input data as a numpy array
        input_data = np.array([[weight, height, duration, speed]])
        
        # Apply the scaler to transform the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction using the loaded model
        prediction = model.predict(input_data_scaled)

        # Render the result in the template
        prediction_on = f'Predicted calories burned based on input: Weight={weight}, Height={height}, Duration={duration}, Speed={speed}'
        prediction_text = f'Calories Burned: {prediction[0]:.2f}'
    except ValueError:
        prediction_on = ''
        prediction_text = 'Invalid input. Please enter numeric values.'
    except Exception as e:
        prediction_on = ''
        prediction_text = f'Error: {str(e)}'

    return render_template('index.html', prediction_on=prediction_on, prediction_text=prediction_text)

# if __name__ == '__main__':
#     app.run(debug=True)
