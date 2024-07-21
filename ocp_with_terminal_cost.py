import tensorflow as tf
from tensorflow.keras.models import load_model
import casadi as ca
import numpy as np
from ocp_double_pendulum_casadi import OcpDoublePendulum

# Load the trained value network
value_network = load_model('value_network.keras', custom_objects={'MeanSquaredError': tf.keras.losses.MeanSquaredError})

# Define a CasADi Callback to wrap the TensorFlow model
class TerminalCostCallback(ca.Callback):
    def __init__(self, name, value_network, opts={}):
        ca.Callback.__init__(self)
        self.value_network = value_network
        self.construct(name, opts)

    def get_n_in(self): return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(4)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(1)

    def eval(self, arg):
        x = np.array(arg[0])
        x_tf = tf.convert_to_tensor(x.reshape(1, -1), dtype=tf.float32)
        terminal_cost = self.value_network(x_tf)
        return [terminal_cost[0, 0].numpy()]

# Create a CasADi external function for the terminal cost
def create_terminal_cost_function(value_network):
    callback = TerminalCostCallback('terminal_cost_cb', value_network)
    x = ca.MX.sym('x', 4)
    cost = ca.Function('cost', [x], [callback.call([x])[0]], ['x'], ['cost'])
    cost_numerical = ca.Function('cost_numerical', [x], [cost(x)], ['x'], ['cost'], {'jit': True, 'jit_options': {'flags': ['-O2']}})
    return cost_numerical

# Define the modified OCP class using the value network
class OcpDoublePendulumWithTerminalCost(OcpDoublePendulum):
    def __init__(self, dt, w_u, terminal_cost_func, u_min=None, u_max=None):
        super().__init__(dt, w_u, u_min, u_max)
        self.terminal_cost_func = terminal_cost_func

    def solve(self, x_init, N):
        self.opti = ca.Opti()
        self.x = self.opti.variable(4, N+1)
        self.u = self.opti.variable(2, N)
        x = self.x
        u = self.u

        self.cost = 0
        for i in range(N):
            self.cost += ca.sumsqr(x[:,i]) + self.w_u * ca.sumsqr(u[:,i])
        terminal_cost = self.terminal_cost_func(x[:, N])
        self.cost += terminal_cost
        self.opti.minimize(self.cost)

        for i in range(N):
            self.opti.subject_to(x[:,i+1] == x[:,i] + self.dt * self.dynamics(x[:,i], u[:,i]))

        if self.u_min is not None and self.u_max is not None:
            for i in range(N):
                self.opti.subject_to(self.opti.bounded(self.u_min, u[:,i], self.u_max))

        self.opti.subject_to(x[:,0] == x_init)

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver('ipopt', opts)

        return self.opti.solve()

if __name__ == "__main__":
    N = 25  # Reduced horizon size
    dt = 0.01
    x_init = np.array([-0.8, 0.0, 0.0, 0.0])
    w_u = 1e-2
    u_min = -2
    u_max = 2

    terminal_cost_func = create_terminal_cost_function(value_network)
    ocp_solver_with_terminal_cost = OcpDoublePendulumWithTerminalCost(dt, w_u, terminal_cost_func, u_min=u_min, u_max=u_max)
    sol_with_terminal_cost = ocp_solver_with_terminal_cost.solve(x_init, N)
    print("Optimal value of x with terminal cost:\n", sol_with_terminal_cost.value(ocp_solver_with_terminal_cost.x))
    print("Optimal value of u with terminal cost:\n", sol_with_terminal_cost.value(ocp_solver_with_terminal_cost.u))

    # Plotting the results
    import matplotlib.pyplot as plt

    time = np.arange(0, (N+1)*dt, dt)
    x_values = sol_with_terminal_cost.value(ocp_solver_with_terminal_cost.x)
    u_values = sol_with_terminal_cost.value(ocp_solver_with_terminal_cost.u)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(time, x_values[0, :], label='q1')
    plt.plot(time, x_values[1, :], label='q2')
    plt.ylabel('Position')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time, x_values[2, :], label='dq1')
    plt.plot(time, x_values[3, :], label='dq2')
    plt.ylabel('Velocity')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time[:-1], u_values[0, :], label='u1')
    plt.plot(time[:-1], u_values[1, :], label='u2')
    plt.xlabel('Time [s]')
    plt.ylabel('Control')
    plt.legend()

    plt.tight_layout()
    plt.show()
