import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

import numpy as np

'''
Comparing Phis; CartPole simulation
'''

a = np.array([0., 33.82, 45.36, 57.96, 73.18, 74.62, 84.24, 94.16, 109.84, 125.86,
                            141.18, 152.36, 164.68, 164.02, 171.32, 179.68, 177.3, 176.12, 182.06, 182.1,
                            188.28, 186.08, 184.98, 185.6, 183.84, 184.52])
adev = np.array([0.54411411, 7.15474668, 11.0882821, 12.44186481, 9.90610418, 11.05348814,
                       12.69373468, 13.73362661, 18.86911763, 20.48166009, 22.64182192, 22.85899385,
                       22.69339111, 20.89388427, 19.89307417, 17.74926477, 19.41159705, 18.48286774,
                       17.05092666, 16.71683283, 17.60795275, 16.58850204, 17.40079309, 17.52940387,
                       17.12182525, 17.48348935]) / 2

e = np.array([ 0.  , 42.85, 42.25, 58.15, 49.15, 63.1 , 74.4 , 84.35, 84.5 ,
       84.35, 82.9 , 69.05, 78.  , 72.05, 93.15, 90.2 , 91.  , 63.9 ,
       83.7 , 90.6 , 84.45, 87.2 , 86.9 , 67.25, 64.65, 78.9 ])
edev = np.array([ 1.26007196,  9.16378743,  8.92258371, 20.50236267, 23.51035676,
       33.14229737, 31.34609226, 39.58227981, 38.5686077 , 41.20900235,
       39.96798719, 34.54535967, 36.93504433, 34.81934627, 41.17744376,
       39.46061327, 39.53985331, 29.60130909, 39.92351281, 39.37619839,
       39.05620149, 38.56296799, 36.54719551, 33.23005981, 31.01890553,
       36.05797415]) / 2

iter = np.arange(26)

plt.plot(iter, a, '-', c='blue', label='7 dimensional \u03A6')
plt.plot(iter, e, ':', c='red', label='4 dimensional \u03A6')
plt.fill_between(iter, a - adev, a + adev, alpha=0.2, color='blue')
plt.fill_between(iter, e - edev, e + edev, alpha=0.2, color='red')
plt.legend(loc='lower right')
#plt.rcParams["font.family"] = "serif"
plt.grid()
plt.title('Average Reward of LFA-NPG Over Policy iterations', fontsize=20)
plt.xlabel('Number of Policy Iterations', fontsize=20)
plt.ylabel('Average Reward', fontsize=20)

plt.ylim(0, 200)
plt.show()
