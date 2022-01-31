import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

import numpy as np


'''
Graphing iterations vs Reward for TRPO, Noise at 0, 10, 15, and 20%; Cartpole simulation
'''

a = np.array([0., 21.504, 27.687, 43.535, 76.352, 108.077, 139.418, 138.661, 162.802,
                            166.119, 157.473, 184.993, 176.291, 165.517, 173.5, 189.473, 176.296, 165.481,
                            186.74, 160.171, 165.558, 162.633, 170.351, 166.581, 159.929, 186.733])
adev = np.array([0., 4.459, 5.908, 8.778, 15.949, 23.78, 32.278, 27.406, 34.833, 33.847,
                       32.143, 38.031, 35.315, 35.45, 34.105, 37.869, 37.47, 35.718, 37.58, 37.839,
                       34.363, 32.196, 34.929, 35.867, 33.779, 38.686]) / 2

b = np.array([  0.   ,  24.357,  26.882,  44.095,  68.994, 103.06 , 152.095,
       170.926, 171.755, 164.239, 170.62 , 164.863, 187.033, 168.915,
       180.5  , 181.276, 179.033, 176.033, 172.943, 191.827, 165.828,
       167.31 , 181.58 , 185.407, 191.1  , 179.889])
bdev = np.array([ 0.   ,  4.954,  5.625, 10.213, 13.954, 23.995, 31.171, 35.612,
       36.657, 35.999, 35.375, 34.248, 37.76 , 35.477, 37.509, 37.855,
       36.773, 36.598, 34.136, 38.717, 35.037, 35.822, 36.607, 37.15 ,
       38.213, 37.452]) / 2

c = np.array([  0.   ,  21.093,  28.467,  40.061,  65.148,  91.83 , 113.992,
       121.068, 130.259, 157.929, 150.939, 142.9  , 155.214, 161.697,
       171.576, 155.473, 173.357, 167.995, 164.78 , 162.476, 184.423,
       181.92 , 173.019, 169.448, 159.557, 171.968])
cdev = np.array([ 0.   ,  4.273,  5.748,  8.768, 13.894, 20.403, 27.392, 25.428,
       27.279, 30.737, 29.656, 30.016, 30.072, 34.451, 34.701, 33.958,
       36.011, 34.965, 37.523, 35.002, 37.471, 36.754, 35.179, 35.167,
       33.164, 35.741]) / 2

d = np.array([  0.   ,  21.736,  27.984,  33.951,  49.775,  94.345, 105.913,
       124.8  , 142.607, 146.694, 142.   , 129.617, 153.89 , 144.788,
       155.854, 152.339, 174.415, 155.638, 172.3  , 168.721, 170.481,
       158.772, 150.542, 164.693, 176.193, 160.051])
ddev = np.array([ 0.   ,  4.288,  5.56 ,  6.739, 10.21 , 21.327, 21.56 , 25.87 ,
       29.89 , 30.299, 29.127, 27.344, 30.688, 32.284, 32.823, 31.875,
       35.739, 32.272, 35.813, 34.839, 34.495, 33.021, 33.134, 33.672,
       35.81 , 33.741]) / 2

iter = np.arange(26)

plt.plot(iter, a, '-', c='blue', label='Noise=0')
plt.plot(iter, b, '--', c='green', label='Noise=10')
plt.plot(iter, c, '-.', c='orange', label='Noise=15')
plt.plot(iter, d, ':', c='red', label='Noise=20')
plt.fill_between(iter, a - adev, a + adev, alpha=0.2, color='blue')
plt.fill_between(iter, b - bdev, b + bdev, alpha=0.2, color='orange')
plt.fill_between(iter, c - cdev, c + cdev, alpha=0.2, color='orange')
plt.fill_between(iter, d - ddev, d + ddev, alpha=0.2, color='red')
plt.legend(loc='lower right')
plt.rcParams["font.family"] = "serif"
plt.grid()
plt.title('Average Reward of TRPO Over Policy iterations', fontsize=22)
plt.xlabel('Number of Policy Iterations', fontsize=20)
plt.ylabel('Average Reward', fontsize=20)

plt.ylim(0, 200)

plt.show()
