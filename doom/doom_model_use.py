import gym
import numpy as np
import scipy.misc as spm
import chainer
from chainer import Function, Variable, optimizers, cuda, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import argparse
import copy
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--name', '-N', default='DoomDefendCenter-v0', type=str,
                    help='game name')
parser.add_argument('--comment', '-c', default='', type=str,
                    help='comment for moniter')                     
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--randomskip', '-rs', default=1, type=int,
                    help='randomskip the frames or not')
parser.add_argument('--n_episode', '-ne', default=120, type=int,
                    help='number of episode to learn')
parser.add_argument('--actionskip', '-as', default=4, type=int,
                    help='number of action repeating')
parser.add_argument('--epsilon', '-ee', default=0.05, type=float,
                    help='the epsilon value')
parser.add_argument('--inputslides', '-sl', type=int, default=4,
                    help='number of input slides')
parser.add_argument('--render', '-r', type=int, default=1,
                    help='rendor or not')
parser.add_argument('--moniter', '-m', type=int, default=0,
                    help='moniter or not')
parser.add_argument('--load', '-l', type=str, default='DoomDefendCenter-v0.model',
                    help='load file')

args = parser.parse_args()


class Preprocess():
    def reward_clip(self, r):
        if r > 0: r = 1.0
        if r < 0: r = -1.0
        return r   

    def gray(self, obs):
        obs_gray = 0.299*obs[:, :, 0] + 0.587*obs[:, :, 1] + 0.114*obs[:, :, 2]
        return obs_gray

    def max(self, obs1, obs2):
        obs_max = np.maximum(obs1, obs2)
        return obs_max

    def downscale(self, obs):
        obs_down = spm.imresize(obs, (84, 84))
        return obs_down

    def one(self, obs):
        processed = self.downscale(self.gray(obs))
        return processed

    def two(self, obs1, obs2):
        processed = self.downscale(self.gray(self.max(obs1, obs2)))
        return processed


class Q(Chain):
    def __init__(self, num_of_actions):
        super(Q, self).__init__(
            l1=L.Convolution2D(4, 32, ksize=8, stride=4),
            l2=L.Convolution2D(32, 64, ksize=4, stride=2),
            l3=L.Convolution2D(64, 64, ksize=3, stride=1),
            l4=L.Linear(3136, 512),
            l5=L.Linear(512, num_of_actions, initialW=np.zeros((num_of_actions, 512),dtype=np.float32))
        )

    def __call__(self, x):
        h_1 = F.relu(self.l1(x / 255.0))
        h_2 = F.relu(self.l2(h_1))
        h_3 = F.relu(self.l3(h_2))
        h_4 = F.relu(self.l4(h_3))
        o = self.l5(h_4)
        return o


class DQN():
    gamma = 0.99
    def __init__(self, gpu, num_of_actions, input_slides):

        self.gpu = gpu
        self.input_slides = input_slides
        self.num_of_actions=num_of_actions

        self.model = Q(num_of_actions)

        if self.gpu >= 0:
            self.model.to_gpu(gpu)                

    def epsilon_greedy(self, s, epsilon):
        s = np.asarray(s.reshape(1, 4, 84, 84), dtype=np.float32)
        if self.gpu >= 0:
            s = cuda.to_gpu(s, device=gpu)
        s = Variable(s)

        q = self.model(s)
        q = q.data[0]
        if self.gpu >= 0:
            q = cuda.to_cpu(q)

        if np.random.rand() < epsilon:
            action = np.random.randint(0, self.num_of_actions)
        else:
            a = np.argmax(q)
            action = np.asarray(a, dtype=np.int8)
        return action

    def action_convert(self, a):
        action = [0] * 43
        if a == 1:
            action[0] = 1
        if a == 2:
            action[14] = 1
        if a == 3:
            action[15] = 1
        return action          

gpu = args.gpu
name = args.name
comment = args.comment
randomskip = args.randomskip
n_episode = args.n_episode
action_skip = args.actionskip
input_slides = args.inputslides
render = args.render
epsilon = args.epsilon
load = args.load
moniter = args.moniter
num_of_actions = 4

env = gym.make(name)
if moniter == 1:
    env.monitor.start('moniter/{}_{}.mon'.format(name, comment))

dqn = DQN(gpu, num_of_actions, input_slides)
serializers.load_npz('network/{}'.format(load), dqn.model)
preprocess = Preprocess()

for i_episode in range(n_episode):
    total_reward = 0
    obs = env.reset()
    
    s = np.zeros((4, 84, 84), dtype=np.uint8)
    s[3] = preprocess.one(obs)

    while (True):
        #gym
        if randomskip == 1:
            if render == 1:env.render()
            a = dqn.epsilon_greedy(s, epsilon)
            a = dqn.action_convert(a)
            obs, reward, done, info = env.step(a)
            obs_processed = preprocess.one(obs)    

        s_prev = copy.deepcopy(s)
        s = np.asanyarray([s[1], s[2], s[3], obs_processed], dtype=np.uint8)

        r = preprocess.reward_clip(reward)
 
        total_reward += reward   
        
        if done:
            print "total reward: {}".format(total_reward)
            break

if moniter == 1:
    env.monitor.close()         

