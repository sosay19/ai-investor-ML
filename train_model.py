
# train_model.py
import tensorflow as tf

# Dummy Data
X_train = [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]
y_train = [1, 0]  # Example labels for binary classification

# Define a simple TensorFlow model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5)

# Save the model
model.save('ml_model/tf_model.h5')
