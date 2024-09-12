import flwr as fl
import tensorflow as tf

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the data (pixel values from 0-255 to 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a very simple model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 images to 1D vectors
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 units (for digits 0-9)
])

# Compile the model
model.compile(
    optimizer='sgd',  # Using Stochastic Gradient Descent (SGD) for simplicity
    loss='sparse_categorical_crossentropy',  # Loss function for classification
    metrics=['accuracy']  # Track accuracy
)

# Define Flower client
class SimpleClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=SimpleClient(),
)
