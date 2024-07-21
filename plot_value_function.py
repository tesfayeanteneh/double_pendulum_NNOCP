import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained value network
value_network = load_model('value_network.keras', custom_objects={'MeanSquaredError': tf.keras.losses.MeanSquaredError})

# Plotting the value function
def plot_value_function(value_network, state_range):
    X1, X2 = np.meshgrid(state_range, state_range)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            state = np.array([X1[i, j], X2[i, j], 0, 0]).reshape(1, -1)
            Z[i, j] = value_network.predict(state)

    plt.figure()
    plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Value Function')
    plt.xlabel('q1')
    plt.ylabel('q2')
    plt.title('Value Function')
    plt.show()

if __name__ == "__main__":
    state_range = np.linspace(-2, 2, 100)
    plot_value_function(value_network, state_range)
