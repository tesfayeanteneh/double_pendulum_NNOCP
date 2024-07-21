import casadi as ca
import numpy as np

def double_pendulum_dynamics():
    # Define symbolic variables
    q1 = ca.SX.sym('q1')
    q2 = ca.SX.sym('q2')
    dq1 = ca.SX.sym('dq1')
    dq2 = ca.SX.sym('dq2')
    u1 = ca.SX.sym('u1')
    u2 = ca.SX.sym('u2')
    x = ca.vertcat(q1, q2, dq1, dq2)
    u = ca.vertcat(u1, u2)

    # Constants
    l1 = 1.0
    l2 = 1.0
    m1 = 1.0
    m2 = 1.0
    g = 9.81

    # Equations of motion
    ddq1 = (l1**2 * l2 * m2 * dq1**2 * ca.sin(-2 * q2 + 2 * q1) + 
            2 * u2 * ca.cos(-q2 + q1) * l1 + 
            2 * (g * ca.sin(-2 * q2 + q1) * l1 * m2 / 2 + 
                 ca.sin(-q2 + q1) * dq2**2 * l1 * l2 * m2 + 
                 g * l1 * (m1 + m2 / 2) * ca.sin(q1) - u1) * l2) / (
            l1**2 * l2 * (m2 * ca.cos(-2 * q2 + 2 * q1) - 2 * m1 - m2))
    
    ddq2 = (-g * l1 * l2 * m2 * (m1 + m2) * ca.sin(-q2 + 2 * q1) - 
            l1 * l2**2 * m2**2 * dq2**2 * ca.sin(-2 * q2 + 2 * q1) - 
            2 * dq1**2 * l1**2 * l2 * m2 * (m1 + m2) * ca.sin(-q2 + q1) + 
            2 * u1 * ca.cos(-q2 + q1) * l2 * m2 + 
            l1 * (m1 + m2) * (ca.sin(q2) * g * l2 * m2 - 2 * u2)) / (
            l2**2 * l1 * m2 * (m2 * ca.cos(-2 * q2 + 2 * q1) - 2 * m1 - m2))
    
    dx = ca.vertcat(dq1, dq2, ddq1, ddq2)
    return ca.Function('f', [x, u], [dx])

if __name__ == "__main__":
    f = double_pendulum_dynamics()
    x0 = np.array([-0.8, 0.0, 0.0, 0.0])
    u0 = np.array([0.0, 0.0])
    print("State derivative at x0, u0:", f(x0, u0))
