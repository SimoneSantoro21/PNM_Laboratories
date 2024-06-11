#%%
import numpy as np
from scipy.optimize import shgo
from scipy.optimize import Bounds

def my_function(x): 
    x1 = x[0] #x1 = TR, x2 = TE/2, the 2 are placeholder values for M0A and M0B
    x2 = x[1]
    return (2*(1-np.exp(-x1/80)*2*np.exp((x2/30)-1))*np.exp(-x2/30) - 2*(1-np.exp(-x1/1000)*2*np.exp((x2/500)-1))*np.exp(-x2/500))

def inverted_function(x):
    if my_function(x) >= 0 :
        return 1/my_function(x)
    else :
        return -1*1/my_function(x)

initial_guess = np.array([200.0, 10.0])

bounds = Bounds([100, 5], [2000, 500])

results = dict()

results['shgo_sobol'] = shgo(inverted_function, bounds, n=1000, iters=5,sampling_method='sobol')

results['shgo_sobol'] #results obtained with this definition of contrast make little sense

# %%
import numpy as np
from scipy.optimize import shgo
from scipy.optimize import Bounds


def my_function(x): 
    x1 = x[0] #x1 = TR, x2 = TE/2, the 2 are placeholder values for M0A and M0B
    x2 = x[1]
    SA = 2*(1-np.exp(-x1/80)*2*np.exp((x2/30)-1))*np.exp(-x2/30)
    SB = 2*(1-np.exp(-x1/1000)*2*np.exp((x2/500)-1))*np.exp(-x2/500)
    AbsSA = SA
    AbsDiff = SA-SB
    if SA < 0 :
        AbsSA = -1*SA
    if AbsDiff < 0 :
        AbsDiff = -1*AbsDiff
    return AbsDiff/AbsSA

def inverted_function(x):
    if my_function(x) >= 0 :
        return 1/my_function(x)

initial_guess = np.array([200.0, 10.0])

bounds = Bounds([100, 5], [10000, 500])

results = dict()

results['shgo_sobol'] = shgo(inverted_function, bounds, n=1000, iters=5,sampling_method='sobol')

results['shgo_sobol']

# %%
