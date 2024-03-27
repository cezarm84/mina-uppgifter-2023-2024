from flask import Flask, request, jsonify, render_template, session, make_response
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib
import jwt
from datetime import datetime, timedelta    
from functools import wraps

# Load the trained Ridge Regression model
ridge_model = joblib.load("random_forest_model.pkl")

app = Flask(__name__)

# Secret key for JWT token
app.config['SECRET_KEY'] = 'SECRET_KEY'# kan generear på flera olika sätt uuid, eller os.urandom...mm


# Decorator for token authentication
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.args.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            payload = jwt.decode(token, app.config['SECRET_KEY'])
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token!'}), 401

        return f(*args, **kwargs)

    return decorated

@app.route('/')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return " redan loggat in" 

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    if username == 'user' and password == '123456':
        session['logged_in'] = True
        # Generate token
        token = jwt.encode({
            'user': username, 
            'exp': datetime.utcnow() + timedelta(minutes=1)
        }, app.config['SECRET_KEY'])
        return jsonify({'token': token}), 200# kolla token på jwt.io  for att dekoda
      
    else:
        # Authentication failed
        return jsonify({'message': 'Authentication failed!'}), 403

@app.route('/predict', methods=['POST'])
@token_required
def predict():
    # Get JSON data from the request body
    data = request.get_json()

    # Check if any of the required fields are missing
    required_fields = ["age", "condition", "distance", "interest_rate", "pool", "rooms"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

    # Extract fråm JSON data
    age = data.get('age')
    condition = data.get('condition')
    distance = data.get('distance')
    interest_rate = data.get('interest_rate')
    pool = data.get('pool')
    rooms = data.get('rooms')

    # Perform prediction using the received parameters
    predicted_price = predict_price(age, condition, distance, interest_rate, pool, rooms)

    # Return the predicted price as JSON response
    return jsonify({'predicted_price': predicted_price}), 200

def predict_price(age, condition, distance, interest_rate, pool, rooms):
    # Scale the features
    features = np.array([[age, condition, distance, interest_rate, pool, rooms]])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Make prediction using the model
    predicted_price = ridge_model.predict(scaled_features)

    return predicted_price[0]

if __name__ == '__main__':
    app.run(debug=True)
