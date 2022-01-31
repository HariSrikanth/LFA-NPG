import numpy as np  # import statements
import gym
from scipy.special import softmax
import matplotlib.pyplot as plt
import math
import time
import random
from sklearn.linear_model import LogisticRegression


class QNPG:  # Code to execute QNPG Code

    def __init__(self, env, noise):  # Initialize our program with all state variables
        self.actdistro = np.arange(3)
        self.env = env
        self.gamma = 0.95
        self.alpha = 0.0001
        self.nu = 1
        self.T = 40
        self.N = 150
        self.d = 18 # dimensions of phi
        self.theta = np.arange(self.d)
        self.bigW = 1000000000000
        self.noise = noise/100

    def AlgOne(self, theta):

        action = self.env.action_space.sample()  # sample action from list of all possible actions
        state = np.zeros(6)

        qhat = 0
        h = 0

        done = False

        def randval(input):
            output = input * random.uniform(1-self.noise, 1+self.noise)
            return output

        while (np.random.choice(range(2), p=[self.gamma, 1 - self.gamma]) == 0) or done:  # while within termination probability

            observation, reward, done, info = self.env.step(action)  # sample the next state and reward from openAI.gym
            state = observation


            pi = self.softmax(state, theta)
            # self.env.render()
            action = np.random.choice(np.arange(3), p=pi)
            qhat += reward  # add rewared gained in this step to cumulative reward
            h += 1  # increase iteration count by 1

            if done:
                self.env.reset()

        self.env.reset()

        return h, state, action, qhat  # once algorithm 1 has been terminated, return iteration count and total reward

    def episodicTimeUp(self, theta):
        print("testing")
        action = self.env.action_space.sample()  # sample action from list of all possible actions
        state = np.arange(6)
        done = False

        qhat = 0
        h = 0

        trackerRwd = 0
        internalElapsed = 0

        def randval(input):
            output = input * random.uniform(1-self.noise, 1+self.noise)
            return output

        while not done:  # no termination probability

            observation, reward, done, info = self.env.step(action)  # sample the next state and reward from openAI.gym
            state = observation

            pi = self.softmax(state, theta)
            # self.env.render()
            action = np.random.choice(np.arange(3), p=pi)
            qhat += reward  # add rewared gained in this step to cumulative reward
            h += 1  # increase iteration count by 1

            if done:
                self.env.reset()
                break

        self.env.reset()

        return qhat, trackerRwd, internalElapsed


    def AlgTwo(self):

        theta = np.arange(self.d)
        aggrwdmat = np.empty(1)
        timemat = np.empty(1)

        maxaggrwd = 0
        maxtheta = np.zeros(self.d)
        totaltime = 0

        outerTime = time.process_time()

        for t in range(self.T):

            aggregateW = np.zeros(self.d)  # track total W
            w = np.zeros(self.d)  # initial value
            aggrwd = 0
            num = 0

            elapsed = 0

            for n in range(self.N):

                start = time.process_time()

                h, state, action, reward = self.AlgOne(theta)  # use AlgOne to return the iterations and the reward
                w = w - ((2 * self.alpha) * (
                        (w.dot(self.phi(state, action)) - reward) * self.phi(state, action)))  # update total W

                while np.linalg.norm(w) > self.bigW:
                    w = w * (self.bigW / np.linalg.norm(w))
                    if np.linalg.norm(w) > self.bigW:
                        break
                    #print("Descaling...Complete")

                aggregateW = aggregateW + w

                end = time.process_time()
                elapsed += (end - start)

                # print(aggregateW)

                if n % self.N == 0:
                    aggrwd += self.episodicTimeUp(theta)[0]
                    print(self.episodicTimeUp(theta)[0])
                    num += 1

                # print(theta)

            aggrwdmat = np.append(aggrwdmat, aggrwd / num)
            timemat = np.append(timemat, time.process_time() - outerTime)
            totaltime += elapsed

            if aggrwd / num > maxaggrwd:
                maxaggrwd = aggrwd / num
                maxtheta = theta

            wt = aggregateW / self.N  # Find the average W

            theta = theta + (self.nu * wt)  # update theta

            itern = np.arange((self.T + 1))
            print("Outer Iteration: " + str(t))

            timemat[0] = 0

        regularizedAggrw, regularizedTimemat = self.reg(aggrwdmat, timemat)

        return maxtheta, aggrwdmat, itern, timemat, totaltime, regularizedAggrw, regularizedTimemat

    def phi(self, state, action):
        phis = np.zeros(self.d)
        if action == 0:
            phis[0:6] = state
        elif action == 1:
            phis[6:12] = state
        else:
            phis[12:18] = state

        return phis

    def softmax(self, state, theta):
        total = np.vstack((np.vstack((self.phi(state, 0), self.phi(state, 1))), self.phi(state, 2)))
        pi = softmax(np.dot(total, theta))
        return pi

    def reg(self, x, y):
        # newx = np.append(x, [0, 10, 20, 190, 200])
        # newy = np.append(y, [0,0,0, 6.5, 7])

        regPredictor = np.poly1d(np.polyfit(x, y, 4))
        neededX = np.arange(-500, -100, step=4)
        y_eq = regPredictor(neededX)

        '''
        myline = np.linspace(1, 200, 50)

        plt.scatter(x, y)
        plt.plot(myline, regPredictor(myline))
        plt.xlim(0, 201)
        plt.ylim(0, 40)
        plt.show()
        '''

        return neededX, y_eq


envi = gym.make('Acrobot-v1')  # Start OpenAI.gym simulation
envi.reset()  # Ensure that simulation is in reset state

agent = QNPG(envi, 0)
'''
print(agent.theta)
iterations = np.arange(1001)
rewards = np.empty(1)
rewarded = 0
for i in range(1000):
    rewarded += agent.episodicTimeUp(np.arange(18))[0]
    print(str(i) + ": " + str(rewarded / (i + 1)))
    rewards = np.append(rewards, (rewarded / (i + 1)))
print(rewarded / 1000)
print(np.shape(rewards))
print(np.shape(iterations))
print(rewards)

plt.scatter(iterations, rewards, s=2, c='black')
plt.title('Average reward of Sampler vs number of Iterations')
plt.xlabel('Iterations')
plt.ylabel('Average Reward')
plt.show()
'''
globalstart = time.process_time()

init_result = agent.AlgTwo()
total_reward = init_result[1]
total_time = init_result[3]
totalRegtime = init_result[6]
value_mat = init_result[1]
time_mat = init_result[3]
regTime_mat = init_result[6]
totalruntime = init_result[4]
regRew = init_result[5]

for i in range(4):
    results = agent.AlgTwo()
    total_reward += results[1]
    total_time += results[3]
    regTime = results[6]
    totalRegtime += regTime
    value_mat = np.vstack((value_mat, results[1]))
    time_mat = np.vstack((time_mat, results[3]))
    regTime_mat = np.vstack((regTime_mat, regTime))
    totalruntime += results[4]

globalend = time.process_time()
globalduration = globalend - globalstart

average_reward = total_reward / 5
average_regTime = totalRegtime / 5
averageruntime = totalruntime / 5
average_reward[0] = -500
value_mat[0, :] = -500

rewarddev = np.std(value_mat, axis=0) / 2
timedev = np.std(regTime_mat, axis=0) / 2

plt.plot(results[2], average_reward, '-', c='purple')
plt.fill_between(results[2], average_reward - rewarddev, average_reward + rewarddev, alpha=0.2)
plt.rcParams["font.family"] = "serif"
plt.grid()
plt.title('Average Reward of QNPG Over Policy iterations')
plt.xlabel('Iterations')
plt.ylabel('Average Reward')
plt.ylim(-500, 0)

plt.show()

plt.plot(regRew, average_regTime, '-', c='purple')
plt.fill_between(regRew, average_regTime - timedev, average_regTime + timedev, alpha=0.2)
plt.rcParams["font.family"] = "serif"
plt.grid()
plt.title('Average Time of QNPG Over Reward')
plt.xlabel('Iterations')
plt.ylabel('Average Time')
plt.ylim(0)

plt.show()

print("The theta that provides the maximum reward: " + str(repr(results[0])))

print("Total average runtime:" + str(averageruntime))
print("The total time for the run: " + str(globalduration))

print(str(repr(average_regTime)))
print(str(repr(average_reward)))
print(str(repr(regRew)))
print(str(repr(timedev)))
print(str(repr(rewarddev)))

# [-5.32552342e+01, -1.06221093e+01, 3.22093019e-02, 6.12411661e+01,
#                                       2.56871717e+00, 4.32815746e+00, 5.32815746e+00, -5.21600256e+01,
#                                       6.67548924e+00, 1.28675946e+01, 6.82121139e+01, 1.68099204e+01,
#                                       1.73614224e+01, 1.83614224e+01]
