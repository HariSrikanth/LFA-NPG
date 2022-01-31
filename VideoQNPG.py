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
        self.actdistro = np.arange(2)
        self.env = env
        self.gamma = 0.95
        self.alpha = 0.1
        self.nu = 0.1
        self.T = 25
        self.N = 150
        self.d = 14  # dimensions of phi
        self.theta = np.arange(self.d)
        self.bigW = 1000000000000
        self.noise = noise / 100

    def AlgOne(self, theta):

        action = self.env.action_space.sample()  # sample action from list of all possible actions
        state = np.zeros(7)

        qhat = 0
        h = 0

        done = False

        def randval(input):
            output = input * random.uniform(0.9, 1.1)
            return output

        while (np.random.choice(range(2),
                                p=[self.gamma, 1 - self.gamma]) == 0) or done:  # while within termination probability

            observation, reward, done, info = self.env.step(action)  # sample the next state and reward from openAI.gym
            state = [observation[0], observation[1], math.sin(observation[2]), math.cos(observation[2]), observation[3],
                     math.sin(observation[3]), math.sin(observation[3])]
            state = [randval(observation[0]), randval(observation[1]), randval(math.sin(observation[2])),
                     randval(math.cos(observation[2])), randval(observation[3]), randval(math.sin(observation[3])),
                     randval(math.sin(observation[3]))]  # for robustness analysis

            pi = self.softmax(state, theta)
            # self.env.render()
            action = np.random.choice(np.arange(2), p=pi)
            qhat += reward  # add rewared gained in this step to cumulative reward
            h += 1  # increase iteration count by 1

            if done:
                self.env.reset()

        self.env.reset()

        return h, state, action, qhat  # once algorithm 1 has been terminated, return iteration count and total reward

    def episodicTimeUp(self, theta):
        print("testing")
        action = self.env.action_space.sample()  # sample action from list of all possible actions
        state = np.arange(7)
        done = False

        qhat = 0
        h = 0

        trackerRwd = 0
        internalElapsed = 0

        def randval(input):
            output = input * random.uniform(1 - self.noise, 1 + self.noise)
            return output

        while not done:  # no termination probability

            observation, reward, done, info = self.env.step(action)  # sample the next state and reward from openAI.gym
            state = [observation[0], observation[1], math.sin(observation[2]), math.cos(observation[2]), observation[3],
                     math.sin(observation[3]), math.sin(observation[3])]
            state = [randval(observation[0]), randval(observation[1]), randval(math.sin(observation[2])),
                     randval(math.cos(observation[2])), randval(observation[3]), randval(math.sin(observation[3])),
                     randval(math.sin(observation[3]))]  # for robustness analysis

            pi = self.softmax(state, theta)
            # self.env.render()
            action = np.random.choice(np.arange(2), p=pi)
            qhat += reward  # add rewared gained in this step to cumulative reward
            h += 1  # increase iteration count by 1

            if done:
                self.env.reset()
                break

        self.env.reset()

        return qhat, trackerRwd, internalElapsed

    def episodicVideoTester(self, theta):
        videoEnv = gym.wrappers.Monitor(gym.make('Acrobot-v1'), './vid', video_callable=lambda episode_id: True,
                                        resume=True)
        videoEnv.reset()

        print("testing")
        action = self.env.action_space.sample()  # sample action from list of all possible actions
        state = np.arange(7)
        done = False

        qhat = 0
        h = 0

        trackerRwd = 0
        internalElapsed = 0

        def randval(input):
            output = input * random.uniform(1 - self.noise, 1 + self.noise)
            return output

        while not done:  # no termination probability

            observation, reward, done, info = videoEnv.step(action)  # sample the next state and reward from openAI.gym
            state = [observation[0], observation[1], math.sin(observation[2]), math.cos(observation[2]), observation[3],
                     math.sin(observation[3]), math.sin(observation[3])]
            state = [randval(observation[0]), randval(observation[1]), randval(math.sin(observation[2])),
                     randval(math.cos(observation[2])), randval(observation[3]), randval(math.sin(observation[3])),
                     randval(math.sin(observation[3]))]  # for robustness analysis

            pi = self.softmax(state, theta)
            action = np.random.choice(np.arange(2), p=pi)
            qhat += reward  # add rewared gained in this step to cumulative reward
            h += 1  # increase iteration count by 1

            if done:
                videoEnv.reset()
                break

        videoEnv.reset()

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
                    # print("Descaling...Complete")

                aggregateW = aggregateW + w

                end = time.process_time()
                elapsed += (end - start)

                # print(aggregateW)

                if n % self.N == 0:
                    aggrwd += self.episodicTimeUp(theta)[0]
                    print(self.episodicTimeUp(theta)[0])
                    num += 1

                    if t % 5 == 0:
                        self.episodicVideoTester(theta)

                # print(theta)

            aggrwdmat = np.append(aggrwdmat, aggrwd / num)
            timemat = np.append(timemat, time.process_time() - outerTime)
            totaltime += elapsed

            if aggrwd / num > maxaggrwd:
                maxaggrwd = aggrwd / num
                maxtheta = theta

            wt = aggregateW / self.N  # Find the average W
            if aggrwd / num < 195:
                theta = theta + (self.nu * wt)  # update theta

            itern = np.arange((self.T + 1))
            print("Outer Iteration: " + str(t))

            timemat[0] = 0

        regularizedAggrw, regularizedTimemat = self.reg(aggrwdmat, timemat)

        return maxtheta, aggrwdmat, itern, timemat, totaltime, regularizedAggrw, regularizedTimemat

    def phi(self, state, action):
        phis = np.zeros(self.d)
        if action == 0:
            phis[0:7] = state
        else:
            phis[7:14] = state

        return phis

    def softmax(self, state, theta):
        total = np.vstack((self.phi(state, 0), self.phi(state, 1)))
        pi = softmax(np.dot(total, theta))
        return pi

    def reg(self, x, y):
        regPredictor = np.poly1d(np.polyfit(x, y, 4))
        neededX = np.arange(200, step=4)
        y_eq = regPredictor(neededX)

        # myline = np.linspace(1, 200, 50)

        # plt.scatter(x, y)
        # plt.plot(myline, regPredictor(myline))
        # plt.xlim(0, 201)
        # plt.ylim(0, 40)
        # plt.show()

        return neededX, y_eq


envi = gym.make('CartPole-v0')  # Start OpenAI.gym simulation
envi.reset()  # Ensure that simulation is in reset state

agent = QNPG(envi, 5)

agent.AlgTwo()



