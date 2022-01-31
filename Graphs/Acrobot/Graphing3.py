import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

import numpy as np


'''
Graphing Iterations vs Reward for QNPG, noise at 0, 10, 15, and 20% deviation; Acrobot simulation
'''

a = np.array([-500. , -500. , -500. , -500. , -500. , -500. , -500. , -483.8,
       -438.6, -468. , -422.4, -491.6, -313. , -364. , -490.6, -253.8,
       -350.6, -326.4, -175.6, -357.8, -337.8, -162.8, -209.4, -261.8,
       -191.2, -245.4, -215.4, -171.4, -159.8, -140.8, -179. , -194.2,
       -207.6, -195.6, -159.4, -174.4, -144.4, -252.6, -168.6, -204.8,
       -183. , -179. , -147. , -146. , -181.8, -191.4, -155.8, -153.2,
       -175.8, -149.4, -128.8, -156.6, -255.8, -205.6, -168.4, -148.6,
       -212.6, -151.2, -172.8, -183.8, -191.2])
adev = np.array([100.22101319,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
        61.4       ,  22.20135131,  52.15898772,   8.4       ,
        58.8226147 ,  54.04387107,   9.4       ,  63.98312277,
        37.93889824,  67.23510988,  65.64190735,  67.75426186,
        53.86315995,  66.91980275,  75.43845173,  71.54550999,
        68.37470292,  57.54771933,  69.87088092,  70.83106663,
        69.12119791,  72.98054535,  70.37087466,  92.1289314 ,
        57.4273454 ,  61.12397238,  70.46204652,  68.54166616,
        71.41330408,  74.56044528,  71.28155442,  70.58753431,
        70.70318239,  68.64153262,  74.0152687 ,  72.37858799,
        64.70672299,  64.55803591,  70.83741949,  72.57575353,
        64.02999297,  69.49863308,  75.05904343,  69.2470938 ,
        75.44905566,  64.09882994,  68.84228352,  71.85429702,
        67.1173599 ,  72.32606722,  66.56831078,  72.35025916,
        66.68088182]) / 2



c = np.array([-500. , -500. , -500. , -500. , -500. , -500. , -483. , -481.2,
       -413.4, -476.8, -410.6, -474.2, -432.8, -391.6, -408.4, -241. ,
       -311.2, -468.6, -342.8, -266.2, -231.6, -369. , -187. , -316. ,
       -156.4, -166.8, -130.8, -189. , -257.6, -182.8, -200.2, -174.8,
       -155.6, -153.2, -146.8, -158.2, -179.6, -160.2, -172.6, -214.2,
       -129. , -163.4, -148. , -159.8, -157.6, -185. , -187.4, -231.2,
       -111.4, -149.2, -184.2, -210. , -121.4, -173.6, -192.4, -226.2,
       -131.8, -113.6, -131.4, -185.4, -141.6])
cdev = np.array([100.63693381,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,  17.        ,  18.8       ,
        67.78318376,  23.2       ,  56.8       ,  25.8       ,
        31.12780108,  51.89450838,  21.8       ,  59.4006734 ,
        77.48199791,   0.        ,  73.66573152,  75.54892455,
        57.01526111,  53.65034949,  65.07918254,  75.96643996,
        69.07575552,  66.59174123,  75.60291   ,  69.54466191,
        83.56171372,  65.42140323,  65.67678433,  67.45739396,
        72.03985008,  69.67739949,  70.45608561,  71.19241533,
        69.30180373,  68.06658505,  68.47159995,  88.06985864,
        75.65487426,  68.50284666,  72.56955284,  68.11637101,
        69.46697057,  71.72140545,  66.67503281,  84.05093694,
        78.47458187,  72.53578427,  64.94412983,  70.89287693,
        76.54867732,  69.98614149,  61.88731696,  58.69633719,
        76.81926842,  78.03268546,  74.52959144,  64.60990636,
        75.15277773]) / 2

d = np.array([-500. , -500. , -490.6, -500. , -500. , -500. , -494.6, -435. ,
       -377.6, -472.2, -388. , -391.8, -442.2, -346.6, -249. , -372.4,
       -302.8, -245.2, -231. , -167.6, -260. , -257.2, -159.8, -244. ,
       -209.2, -191.8, -237.2, -230.6, -166.2, -212.6, -156.2, -190.4,
       -220.2, -191. , -128.6, -184.2, -180.8, -160.2, -167.4, -218.8,
       -179.4, -176. , -176. , -175.6, -166.2, -211.6, -152. , -250.6,
       -176.4, -161. , -161.6, -143.8, -160. , -162.6, -161.6, -217. ,
       -215.4, -160.2, -176.6, -137.8, -205.6])
ddev = np.array([100.2267979 ,   0.        ,   9.4       ,   0.        ,
         0.        ,   0.        ,   5.4       ,   0.        ,
        35.4014124 ,  27.8       ,  73.95295802,  66.2588862 ,
        36.4       ,  43.57591078,  53.52943116,  67.42106496,
        79.80814495,  64.69497662,  62.1760404 ,  71.37128274,
        64.60448901,  51.73142179,  71.18328455,  56.58480361,
        68.24543941,  66.83532   ,  82.95504807,  85.42809842,
        69.12409131,  77.60451018,  71.39985994,  63.51802894,
        63.32661368,  71.44970259,  76.6090073 ,  69.74926523,
        69.81002793,  66.75147938,  71.75681152,  62.32848466,
        61.37507637,  68.04674864,  69.57327073,  66.22491978,
        66.01408941,  87.74713671,  77.26603393,  83.01180639,
        66.89768307,  72.87729413,  69.44393422,  71.60907764,
        73.46114619,  70.58427587,  66.91158345,  57.33027124,
        87.68888185,  71.33400311,  67.77270837,  71.65430901,
        90.81354525]) / 2

e = np.array([-500. , -500. , -500. , -500. , -500. , -398. , -470. , -397.6,
       -458. , -418.4, -379. , -378.8, -361.6, -394.2, -260.4, -372.6,
       -328.4, -245.6, -263. , -215.6, -195.6, -294.8, -198.2, -169.4,
       -257.8, -236.2, -169. , -172.6, -178. , -158.2, -170.4, -167.2,
       -191.8, -155. , -200.2, -160.2, -157.6, -207.8, -185.2, -183.4,
       -172.4, -201.8, -171.6, -164. , -148.4, -167.6, -153.2, -119.2,
       -163. , -184.8, -150.8, -172.8, -152.8, -160.4, -168.8, -180.2,
       -217.2, -154.8, -160.6, -189.8, -156. ])
edev = np.array([100.40353295,   0.        ,   0.        ,   0.        ,
         0.        ,  64.92380149,  30.        ,  43.65844706,
        28.25597282,  50.4       ,  74.46878541,  72.41367274,
        52.9201285 ,  55.34654461,  60.93488328,  42.35256781,
        72.86604147,  84.59338036,  52.95148723,  66.54291848,
        63.21265696,  66.37589924,  67.51399855,  67.42581108,
        58.20532622,  69.04737504,  68.1201879 ,  67.72193146,
        65.80015198,  71.20744343,  66.48864565,  69.30844104,
        63.67369944,  73.28847113,  66.35239257,  70.29395422,
        71.60544672,  58.88089673,  63.87534736,  60.70798959,
        67.20967192,  62.92217415,  66.28770625,  72.75685535,
        71.77854833,  66.65028132,  69.96542003,  76.73239212,
        73.82384439,  62.31259905,  74.28431867,  75.94432171,
        73.32148389,  69.11541073,  70.31742316,  63.16850481,
        86.59422614,  68.18108242,  71.02844501,  67.8092914 ,
        71.55808829]) / 2

iter = np.arange(61)

plt.plot(iter, a, '-', c='blue', label='Noise=0')
#plt.plot(iter, b, '-', c='green', label='Noise=5')
plt.plot(iter, c, '--', c='green', label='Noise=10')
plt.plot(iter, d, '-.', c='orange', label='Noise=15')
plt.plot(iter, e, ':', c='red', label='Noise=20')
plt.fill_between(iter, a - adev, a + adev, alpha=0.2, color='blue')
#plt.fill_between(iter, b - bdev, b + bdev, alpha=0.2, color='green')
plt.fill_between(iter, c - cdev, c + cdev, alpha=0.2, color='green')
plt.fill_between(iter, d - ddev, d + ddev, alpha=0.2, color='orange')
plt.fill_between(iter, e - edev, e + edev, alpha=0.2, color='red')
plt.legend(loc='lower right')
plt.rcParams["font.family"] = "serif"
plt.grid()
plt.title('Average Reward of LFA-NPG Over Policy iterations', fontsize=20)
plt.xlabel('Number of Policy Iterations', fontsize=20)
plt.ylabel('Average Reward', fontsize=20)

plt.ylim(-500, 0)

plt.show()