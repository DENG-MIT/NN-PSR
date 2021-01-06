import cantera as ct
import numpy as np
import scipy.integrate

class ReactorOde(object):
    def __init__(self, gas_ini):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorOde object.
        self.gas = gas_ini
        self.P = gas_ini.P

        # add additional intial conditions
        # tres, ta, q, etc.
        self.yin = gas_ini.Y
        self.tin = gas_ini.T
        self.Q = 8e2
        self.ta = 760
        self.tres = 1.0

    def __call__(self, t, y):
        """the ODE function, y' = f(t,y) """

        # State vector is [T, Y_1, Y_2, ... Y_K]
        self.gas.set_unnormalized_mass_fractions(y[1:])
        self.gas.TP = y[0], self.P
        rho = self.gas.density

        wdot = self.gas.net_production_rates
        dYdt = (self.yin-self.gas.Y)/self.tres + wdot*self.gas.molecular_weights/rho
        dTdt = (self.tin-self.gas.T)/self.tres + \
               self.Q*(self.ta-self.gas.T)/(rho*self.gas.cp) - \
               (np.dot(self.gas.partial_molar_enthalpies,wdot)/(rho*self.gas.cp))


        return np.hstack((dTdt, dYdt))


gas = ct.Solution('gri30.xml')
gas_ini = ct.Solution('gri30.xml')
# Initial condition
# P = 25*133.322  #Pa
# gas.TPX = 800, P, 'H2:0.5,CO:49.5,O2:50'
P = 101325  #Pa
gas.TPX = 300, P, 'H2:9.5023,CO:1.7104,CH4:5.7014,O2:17.0090,N2:66.0769'

i_var = [gas.species_index(s) for s in ['H2', 'CO', 'CH4', 'O2', 'N2', 'CO2']]
Yin = gas.Y[i_var]
TYin = np.append(gas.T, Yin)
# TYin = np.append(TYin, states.T[ind_start])
np.savetxt('TYin.txt', TYin)

gas.equilibrate('HP', solver='gibbs')
y0 = np.hstack((gas.T, gas.Y))
gas_ini.TPX = 300, P, 'H2:9.5023,CO:1.7104,CH4:5.7014,O2:17.0090,N2:66.0769'

# Set up objects representing the ODE and the solver
ode = ReactorOde(gas_ini)
solver = scipy.integrate.ode(ode)
solver.set_integrator('vode', method='bdf', with_jacobian=True)
solver.set_initial_value(y0, 0.0)

# Integrate the equations, keeping T(t) and Y(k,t)
t_end = 3
states = ct.SolutionArray(gas, 1, extra={'t': [0.0]})
dt = 1e-3
while solver.successful() and solver.t < t_end:
    solver.integrate(solver.t + dt)
    gas.TPY = solver.y[0], P, solver.y[1:]
    states.append(gas.state, t=solver.t)
    print("integration time=%s, reactor Temperautre=%s" %(solver.t,gas.T))

print("PSR integration has done!")

ind_start = 2500
ind_stop = 2600

Y = states.Y[:, i_var].T

nodedata = np.vstack((states.t - states.t[ind_start], states.T, Y)).T #states.P / ct.one_atm, 
np.savetxt('data_T.txt', nodedata[ind_start:ind_stop, :])

# Plot the results
try:
    import matplotlib.pyplot as plt
    L1 = plt.plot(states.t, states.T, color='r', label='T', lw=2)
    plt.xlim([2.50, 3])
    plt.ylim([1200, 1300])
    plt.xlabel('time (s)')
    plt.ylabel('Temperature (K)')
    plt.twinx()
    L2 = plt.plot(states.t, states('CO').Y, label='CO', lw=2)
    # plt.ylim([0, 1e-4])
    plt.ylabel('Mass Fraction')
    plt.legend(L1+L2, [line.get_label() for line in L1+L2], loc='best')
    plt.show()
except ImportError:
    print('Matplotlib not found. Unable to plot results.')