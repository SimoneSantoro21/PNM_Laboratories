#%%

import numpy as np
from scipy.optimize import shgo
from scipy.optimize import Bounds
import matplotlib.pyplot as plt

def my_function(x): 
    x1 = x[0] #x1 = TR, x2 = TE/2, the 2 are placeholder values for M0A and M0B
    x2 = x[1]
    return -1*np.abs(((7.57e05)*(1-np.exp(-x1/78.9530)*(2*np.exp(x2/35.8829)-1))*np.exp(-2*x2/35.8829) - (9.03e05)*(1-np.exp(-x1/1010.5)*(2*np.exp(x2/356.7618)-1))*np.exp(-2*x2/356.7618)))

initial_guess = np.array([1000.0, 5])

bounds = Bounds([100, 2.5], [1200, 10])

def inverted(x):
    return 1/inverted(x)

results = dict()

results['shgo_sobol'] = shgo(my_function, bounds, sampling_method='simplicial',n=100000, iters =5)

results['shgo_sobol'] 

x=results['shgo_sobol'].xl
y = results['shgo_sobol'].funl
print(x)
print(y)
print(np.shape(y))
plt.plot(y)

# %%
import numpy as np
from scipy.optimize import shgo
from scipy.optimize import Bounds
import matplotlib.pyplot as plt

def my_function(x1,x2): 
#x1 = TR, x2 = TE/2, the 2 are placeholder values for M0A and M0B
    return np.abs(((7.57e05)*(1-np.exp(-x1/78.9530)*(2*np.exp(x2/35.8829)-1))*np.exp(-2*x2/35.8829) - (9.03e05)*(1-np.exp(-x1/1010.5)*(2*np.exp(x2/356.7618)-1))*np.exp(-2*x2/356.7618)))

y = np.linspace(75,300,num=300)
x = np.linspace(1000,1010,num=4000)

X1, X2 = np.meshgrid(x,y)
results = my_function(X1, X2)

max_value = np.max(results)
max_index = np.unravel_index(np.argmax(results), results.shape)
max_x1, max_x2 = x[max_index[1]], y[max_index[0]]
            
print(max_value)
print(max_x1)
print(max_x2)   


# %%
