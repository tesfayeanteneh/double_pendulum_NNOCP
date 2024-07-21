import numpy as np
import casadi as ca
from double_pendulum_casadi import double_pendulum_dynamics

class OcpDoublePendulum:
    def __init__(self, dt, w_u, u_min=None, u_max=None):
        self.dt = dt
        self.w_u = w_u
        self.u_min = u_min
        self.u_max = u_max
        self.dynamics = double_pendulum_dynamics()

    def solve(self, x_init, N):
        self.opti = ca.Opti()
        self.x = self.opti.variable(4, N+1)
        self.u = self.opti.variable(2, N)
        x = self.x
        u = self.u

        self.cost = 0
        for i in range(N):
            self.cost += ca.sumsqr(x[:,i]) + self.w_u * ca.sumsqr(u[:,i])
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

def generate_ocp_samples(num_samples, dt, w_u, N):
    ocp = OcpDoublePendulum(dt, w_u, u_min=-2, u_max=2)
    X = []
    Y = []
    for _ in range(num_samples):
        x0 = np.random.uniform(low=-2, high=2, size=(4,))
        try:
            sol = ocp.solve(x0, N)
            J_x0 = sol.value(ocp.cost)
            X.append(x0)
            Y.append(J_x0)
        except RuntimeError:
            continue
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    dt = 0.01
    w_u = 1e-2
    N = 50
    num_samples = 1000

    X, Y = generate_ocp_samples(num_samples, dt, w_u, N)
    np.save('X_ocp_samples.npy', X)
    np.save('Y_ocp_samples.npy', Y)
