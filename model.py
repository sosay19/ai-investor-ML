
# model.py
import tensorflow as tf

# Load the TensorFlow model
model = tf.keras.models.load_model('ml_model/tf_model.h5')

def predict(data):
    prediction = model.predict(data)
    return prediction.tolist()
