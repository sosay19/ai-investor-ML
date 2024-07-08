from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load the TensorFlow model
model = tf.keras.models.load_model('ml_model/tf_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()

    # Ensure that data is a list of features
    if not isinstance(data, list):
        return jsonify({'error': 'Invalid input format, expected a list of features'}), 400

    # Make a prediction
    investment_features = [data]  # Wrap data in a list
    prediction = model.predict(investment_features).tolist()
    
    # Example response structure
    response = {
        'prediction': prediction[0][0]  # Example prediction output
    }
    print(response)
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000)
