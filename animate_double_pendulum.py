import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from ocp_with_terminal_cost import OcpDoublePendulumWithTerminalCost, terminal_cost_net
from double_pendulum_casadi import double_pendulum_dynamics

# Function to animate the double pendulum
def animate_double_pendulum(sol, dt, filename="double_pendulum.mp4"):
    fig, ax = plt.subplots()
    x_values = sol.value(ocp_solver_with_terminal_cost.x)

    def update(num):
        ax.clear()
        ax.plot([0, np.sin(x_values[0, num])], [0, -np.cos(x_values[0, num])], marker='o')
        ax.plot([np.sin(x_values[0, num]), np.sin(x_values[0, num]) + np.sin(x_values[1, num])],
                [-np.cos(x_values[0, num]), -np.cos(x_values[0, num]) - np.cos(x_values[1, num])], marker='o')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_title(f'Time: {num * dt:.2f}s')

    ani = animation.FuncAnimation(fig, update, frames=range(x_values.shape[1]), repeat=False)
    ani.save(filename, writer='ffmpeg', fps=30)

if __name__ == "__main__":
    N = 25  # Reduced horizon size
    dt = 0.01
    x_init = np.array([-0.8, 0.0, 0.0, 0.0])
    w_u = 1e-2
    u_min = -2
    u_max = 2

    ocp_solver_with_terminal_cost = OcpDoublePendulumWithTerminalCost(dt, w_u, terminal_cost_net=terminal_cost_net, u_min=u_min, u_max=u_max)
    sol_with_terminal_cost = ocp_solver_with_terminal_cost.solve(x_init, N)

    animate_double_pendulum(sol_with_terminal_cost, dt)
