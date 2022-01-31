import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

import numpy as np

'''
Graphing the iterations vs Reward for QNPG, TRPO, and PPO on the Acrobot Sim
'''


average_rewardT = np.array([0., 21.504, 27.687, 43.535, 76.352, 108.077, 139.418, 138.661, 162.802,
                            166.119, 157.473, 184.993, 176.291, 165.517, 173.5, 189.473, 176.296, 165.481,
                            186.74, 160.171, 165.558, 162.633, 170.351, 166.581, 159.929, 186.733])
rewarddevT = np.array([0., 4.459, 5.908, 8.778, 15.949, 23.78, 32.278, 27.406, 34.833, 33.847,
                       32.143, 38.031, 35.315, 35.45, 34.105, 37.869, 37.47, 35.718, 37.58, 37.839,
                       34.363, 32.196, 34.929, 35.867, 33.779, 38.686]) / 2

average_rewardQ = np.array([0., 33.82, 45.36, 57.96, 73.18, 74.62, 84.24, 94.16, 109.84, 125.86,
                            141.18, 152.36, 164.68, 164.02, 171.32, 179.68, 177.3, 176.12, 182.06, 182.1,
                            188.28, 186.08, 184.98, 185.6, 183.84, 184.52])
rewarddevQ = np.array([0.54411411, 7.15474668, 11.0882821, 12.44186481, 9.90610418, 11.05348814,
                       12.69373468, 13.73362661, 18.86911763, 20.48166009, 22.64182192, 22.85899385,
                       22.69339111, 20.89388427, 19.89307417, 17.74926477, 19.41159705, 18.48286774,
                       17.05092666, 16.71683283, 17.60795275, 16.58850204, 17.40079309, 17.52940387,
                       17.12182525, 17.48348935]) / 2
iter = np.arange(26)

average_rewardP = np.array([0., 20.37435501, 26.14182331, 40.74484127,
                            68.11025641, 108.95, 138.68, 143.15,
                            157.6, 151.16666667, 159.9, 162.86666667,
                            166.4, 190.4, 158.33333333, 164.23333333,
                            186.2, 194.6, 197., 189.5,
                            176.23333333, 189.23333333, 169.26666667, 189.1,
                            176.16666667, 172.4])
rewarddevP = np.array([0., 4.28510804, 5.36800313, 9.53769239, 20.82265825,
                       30.29211944, 34.94099884, 33.36631701, 33.28190199, 33.16112181,
                       34.20386593, 36.95552883, 37.15521498, 38.731899, 36.59423148,
                       36.61972753, 38.91580656, 39.78454474, 39.6092161, 38.97896869,
                       35.7151384, 38.63596482, 36.85925182, 38.74028911, 37.11829438,
                       38.4091135]) / 2

plt.plot(iter, average_rewardQ, '-', c='purple', label='LFA-NPG')
plt.plot(iter, average_rewardT, ':', c='red', label='TRPO')
plt.plot(iter, average_rewardP, '--', c='orange', label='PPO')
plt.fill_between(iter, average_rewardT - rewarddevT, average_rewardT + rewarddevT, alpha=0.2, color='red')
plt.fill_between(iter, average_rewardQ - rewarddevQ, average_rewardQ + rewarddevQ, alpha=0.2, color='purple')
plt.fill_between(iter, average_rewardP - rewarddevP, average_rewardP + rewarddevP, alpha=0.2, color='orange')
plt.legend(loc='lower right')
plt.rcParams["font.family"] = "serif"
plt.grid()
plt.title('Average Reward over Policy iterations', fontsize=25)
plt.xlabel('Number of Policy Iterations', fontsize=20)
plt.ylabel('Average Reward', fontsize=20)
plt.ylim(0, 200)


plt.show()
