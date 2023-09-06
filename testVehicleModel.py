import pickle
import numpy as np
from scipy.integrate import solve_ivp
import scipy
import matplotlib.pyplot as plt
from scipy.io import savemat

WB = 2.7
#A_MAX = 6.47
A_MAX = 20

"""filehandler = open('test.obj', 'rb')
data = pickle.load(filehandler)

left = np.asarray(data['left'])
right = np.asarray(data['right'])

plt.plot(left[:, 0], left[:, 1])
plt.plot(right[:, 0], right[:, 1])
plt.axis('equal')
plt.show()"""

filehandler = open('/media/niklas/FLASHDEVICE/computationTime.obj', 'rb')
#filehandler = open('/home/niklas/Documents/Work/Projekte/MotionPlanner/results/CARLAexperimentsNew/run18/computationTime.obj', 'rb')
#filehandler = open('/media/niklas/FLASHDEVICE/trajectories/trajectory4.obj', 'rb')
#filehandler = open('trajectory_0steer_1acc.obj', 'rb')
data = pickle.load(filehandler)
x = np.asarray(data['x'])
u = np.asarray(data['u'])
t = np.asarray(data['t'])
u[:, 0] = np.maximum(-1, np.minimum(u[:, 0], 1))
u[:, 1] = np.maximum(-1, np.minimum(u[:, 1], 1))

savemat('trajectory.mat', {'x': x, 'u': u, 't': t})

v = x[:, 2]
dv = np.diff(v)/np.diff(t)
a = np.maximum(-1, np.minimum(u[:, 0], 1))


ind = np.where(x[:, 2] > 1e-3)[0][0]
x = x[ind:, :]
t = t[ind:]
t = t - t[0]
u = u[ind-1:, :]

tFinal = t[-1]
ode = lambda t, x, u1, u2: [x[2] * np.cos(x[3]) - WB/2 * np.sin(x[3]) * x[2] * np.tan(u2) / WB,
                            x[2] * np.sin(x[3]) + WB/2 * np.cos(x[3]) * x[2] * np.tan(u2) / WB, u1 * A_MAX - 4*x[2], x[2] * np.tan(u2) / WB]
sol = solve_ivp(ode, [0, tFinal], x[0, :], args=(u[0, 0], u[0, 1]), dense_output=True)
t_ = np.linspace(0, tFinal, 101)
x_ = sol.sol(t_)

plt.plot(t, x[:, 2])
plt.plot(t_, x_[2, :], 'r')
plt.show()


plt.plot(x[:, 0], x[:, 1])
plt.plot(x_[0, :], x_[1, :], 'r')
plt.show()

test = 1