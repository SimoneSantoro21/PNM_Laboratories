#%%

import numpy as np
from scipy.optimize import shgo
from scipy.optimize import Bounds
import matplotlib.pyplot as plt

def my_function(x): 
    x1 = x[0] #x1 = TR, x2 = TE/2, the 2 are placeholder values for M0A and M0B
    x2 = x[1]
    return ((7.57e05)*(1-np.exp(-x1/78.9530)*(2*np.exp(x2/35.8829)-1))*np.exp(-2*x2/35.8829) - (9.03e05)*(1-np.exp(-x1/1010.5)*(2*np.exp(x2/356.7618)-1))*np.exp(-2*x2/356.7618))

def inverted_function(x):
    if my_function(x) >= 0 :
        return 1/my_function(x)
    else :
        return -1*1/my_function(x)

initial_guess = np.array([100.0, 5])

bounds = Bounds([100, 2.5], [1300, 7.5])

results = dict()

results['shgo_sobol'] = shgo(inverted_function, bounds, sampling_method='sobol',n=10000, iters =30)

results['shgo_sobol'] 

x=results['shgo_sobol'].xl
y = results['shgo_sobol'].funl
print(x)
print(y)
print(np.shape(y))
plt.plot(y)

# %%
