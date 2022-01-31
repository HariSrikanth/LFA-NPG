import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

import numpy as np


'''
Comparing Phis; Acrobot simulation
'''

a = np.array([-500. , -500. , -500. , -500. , -500. , -500. , -500. , -500. ,
       -500. , -500. , -500. , -500. , -474. , -353.2, -367.6, -386.4,
       -250.6, -214. , -242.8, -221.8, -265.6, -230.2, -273. , -201.4,
       -146.6, -150.6, -164.8, -183.8, -164.6, -134.8, -185.2, -149.2,
       -210.8, -153.4, -179.4, -145.2, -252.6, -179.4, -136.2, -171.2,
       -268. ])
adev = np.array([100.28769572,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
        26.        ,  75.03998934,  52.51913937,  46.8363107 ,
        56.63338238,  57.7255576 ,  79.72490201,  56.4999115 ,
        65.23143414,  82.30820129,  53.8364189 ,  62.99158674,
        70.87383156,  72.2360021 ,  71.37828802,  69.58692406,
        67.90611165,  73.14465121,  61.31035802,  69.36094002,
        94.34564113,  71.44018477,  67.81474766,  70.20227917,
        84.24571206,  69.17542916,  72.87043296,  67.18288473,
        57.37543028]) / 2

e = np.array([-500. , -499.2, -500. , -500. , -447.2, -472. , -457.4, -366. ,
       -349.8, -272. , -246.6, -279.4, -205.8, -168.2, -218.4, -169.8,
       -176.8, -168.4, -141. , -160.8, -206. , -157.4, -224.4, -192.2,
       -139.4, -244.8, -138.4, -238.8, -179.2, -177.2, -180.4, -162.4,
       -179.8, -215.6, -157.2, -187.2, -153.6, -177.2, -186.6, -196.2,
       -180.2])
edev = np.array([100.22613223,   0.        ,   0.        ,   0.        ,
        52.8       ,  28.        ,  42.6       ,  68.5886288 ,
        91.98173732,  75.14891882,  61.9559521 ,  71.08755165,
        72.92503   ,  65.96923525,  69.19364133,  64.94582358,
        64.32184699,  69.81504136,  71.9189822 ,  69.82721532,
        68.08230313,  71.27580235,  62.09798708,  62.75396402,
        73.07024018,  84.93079536,  72.84119713,  75.84655562,
        67.55767906,  68.99739126,  62.38878104,  70.84800632,
        72.42554798,  68.9670936 ,  69.36382342,  66.23246938,
        71.63002164,  62.18970976,  66.53314963,  66.0284787 ,
        63.6278241 ]) / 2

iter = np.arange(41)

plt.plot(iter, e, '-', c='blue', label='7 dimensional \u03A6')
plt.plot(iter, a, ':', c='red', label='6 dimensional \u03A6')

plt.fill_between(iter, a - adev, a + adev, alpha=0.2, color='red')
plt.fill_between(iter, e - edev, e + edev, alpha=0.2, color='blue')
plt.legend(loc='lower right')
plt.rcParams["font.family"] = "serif"
plt.grid()
plt.title('Average Reward of LFA-NPG Over Policy iterations', fontsize=20)
plt.xlabel('Number of Policy Iterations', fontsize=20)
plt.ylabel('Average Reward', fontsize=20)

plt.ylim(-500, 0)

plt.show()
