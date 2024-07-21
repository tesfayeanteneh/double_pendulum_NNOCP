import tensorflow as tf
from tensorflow.keras import layers, optimizers
import numpy as np
from ocp_double_pendulum_casadi import OcpDoublePendulum
import multiprocessing as mp

def solve_ocp_for_initial_state(ocp_solver, x0):
    try:
        sol = ocp_solver.solve(x0, N=50)
        J_x0 = sol.value(ocp_solver.cost)
        return x0, J_x0
    except RuntimeError:
        return None

def generate_sample_data(num_samples, ocp_solver, num_workers=4):
    X = []
    Y = []
    pool = mp.Pool(processes=num_workers)
    initial_states = [np.random.uniform(low=-2, high=2, size=(4,)) for _ in range(num_samples)]
    
    results = pool.starmap(solve_ocp_for_initial_state, [(ocp_solver, x0) for x0 in initial_states])
    
    for result in results:
        if result is not None:
            x0, J_x0 = result
            X.append(x0)
            Y.append(J_x0)
    
    pool.close()
    pool.join()
    
    return np.array(X), np.array(Y)

def get_value_network(input_shape):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(16, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model

if __name__ == "__main__":
    ocp_solver = OcpDoublePendulum(dt=0.01, w_u=1e-2, u_min=-2, u_max=2)
    X_train, Y_train = generate_sample_data(num_samples=1000, ocp_solver=ocp_solver, num_workers=4)

    value_network = get_value_network(input_shape=(4,))
    value_network.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=2)

    value_network.save('value_network.keras')
