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

parser = argparse.ArgumentParser()
parser.add_argument('--name', '-N', default='Pong-v0', type=str,
                    help='game name')
parser.add_argument('--comment', '-c', default='', type=str,
                    help='comment to distinguish output')                    
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--n_episode', '-n', default=10**6, type=int,
                    help='number of episode to learn')
parser.add_argument('--actionskip', '-as', default=4, type=int,
                    help='number of action repeating')
parser.add_argument('--update', '-u', default=16, type=int,
                    help='update freaquency')
parser.add_argument('--targetupdate', '-t', default=10**4, type=int,
                    help='target update freaquency')
parser.add_argument('--save', '-s', default=10**4, type=int,
                    help='save freaquency')
parser.add_argument('--eval', '-e', default=10, type=int,
                    help='evaluation freaquency')
parser.add_argument('--initial', '-i', default=10**4, type=int,
                    help='nimber of initial exploration')
parser.add_argument('--epsilon_decrease', '-ep', default=1.0/10**6, type=float,
                    help='initial epsilon')
parser.add_argument('--batchsize', '-b', type=int, default=32,
                    help='learning minibatch size')
parser.add_argument('--memorysize', '-m', type=int, default=10**5,
                    help='replay memory size')
parser.add_argument('--inputslides', '-sl', type=int, default=4,
                    help='number of input slides')
parser.add_argument('--render', '-r', type=int, default=0,
                    help='rendor or not')

args = parser.parse_args()


class Preprocess():
    def reward_clip(self, r):
        if r > 0: r = 1
        if r < 0: r = -1
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

    def start(self, obs):
        processed = self.downscale(self.gray(obs))
        return processed

    def step(self, obs1, obs2):
        processed = self.downscale(self.gray(self.max(obs1, obs2)))
        return processed


class Q(Chain):
    def __init__(self, num_of_actions):
        super(Q, self).__init__(
            l1=L.Convolution2D(4, 32, ksize=8, stride=4),
            l2=L.Convolution2D(32, 64, ksize=4, stride=2),
            l3=L.Convolution2D(64, 64, ksize=3, stride=1),
            l4=L.Linear(3136, 512),
            l5=L.Linear(512, num_of_actions)
        )

    def __call__(self, x):
        h_1 = F.relu(self.l1(x))
        h_2 = F.relu(self.l2(h_1))
        h_3 = F.relu(self.l3(h_2))
        h_4 = F.relu(self.l4(h_3))
        o = self.l5(h_4)
        return o


class DQN():
    gamma = 0.99
    def __init__(self, gpu, num_of_actions, memory_size, input_slides, batch_size):

        self.gpu = gpu
        self.memory_size = memory_size
        self.input_slides = input_slides
        self.batch_size = batch_size
        self.num_of_actions=num_of_actions

        self.replay_memory = [np.zeros((self.memory_size, self.input_slides, 84, 84), dtype=np.uint8),
                              np.zeros(self.memory_size, dtype=np.uint8),
                              np.zeros((self.memory_size, 1), dtype=np.int8),
                              np.zeros((self.memory_size, self.input_slides, 84, 84), dtype=np.uint8),
                              np.zeros((self.memory_size, 1), dtype=np.bool)]

        self.model = Q(num_of_actions)
        self.target_model = copy.deepcopy(self.model)

        if self.gpu == 1:
            self.model.to_gpu()
            self.target_model.to_gpu()

        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.01)
        self.optimizer.setup(self.model)                 

    def update(self, total_step):
        if total_step < self.memory_size:
            index = np.random.randint(0, total_step, self.batch_size)
        else:
            index = np.random.randint(0, self.memory_size, self.batch_size)

        s_prev_replay = np.ndarray(shape=(self.batch_size, 4, 84, 84), dtype=np.float32)
        a_replay = np.ndarray(shape=(self.batch_size, 1), dtype=np.uint8)
        r_replay = np.ndarray(shape=(self.batch_size, 1), dtype=np.float32)
        s_replay = np.ndarray(shape=(self.batch_size, 4, 84, 84), dtype=np.float32)
        done_replay = np.ndarray(shape=(self.batch_size, 1), dtype=np.bool)

        for i in xrange(self.batch_size):
            s_prev_replay[i] = np.asarray(self.replay_memory[0][index[i]], dtype=np.float32)
            a_replay[i] = self.replay_memory[1][index[i]]
            r_replay[i] = self.replay_memory[2][index[i]]
            s_replay[i] = np.array(self.replay_memory[3][index[i]], dtype=np.float32)
            done_replay[i] = self.replay_memory[4][index[i]]

        if self.gpu == 1:
            s_prev_replay = cuda.to_gpu(s_prev_replay)
            s_replay = cuda.to_gpu(s_replay)

        self.model.zerograds()
        loss = self.compute_loss(s_prev_replay, a_replay, r_replay, s_replay, done_replay)
        loss.backward()
        self.optimizer.update()  

    def compute_loss(self, s_prev, a, r, s, done):
        s_prev = Variable(s_prev)
        s = Variable(s)

        q = self.model(s_prev)
        q_data = q.data
        if self.gpu == 1:
            q_data = cuda.to_cpu(q_data)

        q_target = self.target_model(s)
        q_target_data = q_target.data
        if self.gpu == 1:
            q_target_data = cuda.to_cpu(q_target_data)
        max_q_target = np.asarray(np.amax(q_target_data, axis=1), dtype=np.float32)

        target = np.asanyarray(copy.deepcopy(q_data), dtype=np.float32)
        for i in range(self.batch_size):
            if done[i][0] is True:
                tmp = r[i]
            else:
                tmp = r[i] + self.gamma * max_q_target[i]
            target[i, a[i]] = tmp

        if self.gpu == 1:
            target = cuda.to_gpu(target)
        td = Variable(target) - q
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)

        zero = np.zeros((self.batch_size, self.num_of_actions), dtype=np.float32)
        if self.gpu == 1:
            zero = cuda.to_gpu(zero)
        zero =  Variable(zero)    
        loss = F.mean_squared_error(td_clip, zero)
        return loss

    def epsilon_greedy(self, s, epsilon):
        s = np.asarray(s.reshape(1, 4, 84, 84), dtype=np.float32)
        if self.gpu == 1:
            s = cuda.to_gpu(s)
        s = Variable(s)

        q = self.model(s)
        q = q.data[0]
        if self.gpu == 1:
            q = cuda.to_cpu(q)

        if np.random.rand() < epsilon:
            action = np.random.randint(0, self.num_of_actions)
            #print("RANDOM : " + str(action))
        else:
            a = np.argmax(q)
            #print("GREEDY  : " + str(a))
            action = np.asarray(a, dtype=np.int8)
            #print(q)
        return action

    def store(self, total_step, s_prev, a, r, s, done):
        index = total_step % self.memory_size
        self.replay_memory[0][index] = s_prev
        self.replay_memory[1][index] = a
        self.replay_memory[2][index] = r
        self.replay_memory[3][index] = s
        self.replay_memory[4][index] = done  

    def target_update(self):
        self.target_model = copy.deepcopy(self.model)


gpu = args.gpu
name = args.name
comment = args.comment
n_episode = args.n_episode
action_skip = args.actionskip
update_freq = args.update
target_update_freq = args.targetupdate
save_freq = args.save
evl_freq = args.eval
initial_exploration = args.initial
epsilon_decrease = args.epsilon_decrease
memory_size = args.memorysize
input_slides = args.inputslides
batch_size = args.batchsize
render = args.render
epsilon = 1.0
total_step = 0

env = gym.make(name)
dqn = DQN(gpu, env.action_space.n, memory_size, input_slides, batch_size)
preprocess = Preprocess()

def evaluation():
    evaluate_epsilon = 0.1
    step = 0
    total_reward = 0
    obs = env.reset()
    
    s = np.zeros((4, 84, 84), dtype=np.uint8)
    s[3] = preprocess.start(obs)

    while (True):
        if render == 1:
            env.render()
        if step % action_skip == 0:
            a = dqn.epsilon_greedy(s, evaluate_epsilon)

        obs_prev = copy.copy(obs)
        obs, reward, done, info = env.step(a)
        obs_processed = preprocess.step(obs_prev, obs)

        s_prev = copy.deepcopy(s)
        s = np.asanyarray([s[1], s[2], s[3], obs_processed], dtype=np.uint8)

        r = preprocess.reward_clip(reward)

        step += 1 
        total_reward += reward
        
        if done:
            f = open("evaluation/{}({}).txt".format(name, comment), "a")
            f.write(str(i_episode) + "," + str(total_reward) + ',' + str(step) +"\n")
            f.close()
            break


for i_episode in range(n_episode):
    if i_episode % evl_freq == 0:
        print 'Evaluation'
        evaluation()

    step = 0
    total_reward = 0
    obs = env.reset()
    
    s = np.zeros((4, 84, 84), dtype=np.uint8)
    s[3] = preprocess.start(obs)

    while (True):
        if render == 1:
            env.render()
        if step % action_skip == 0:
            a = dqn.epsilon_greedy(s, epsilon)

        obs_prev = copy.copy(obs)
        obs, reward, done, info = env.step(a)
        obs_processed = preprocess.step(obs_prev, obs)

        s_prev = copy.deepcopy(s)
        s = np.asanyarray([s[1], s[2], s[3], obs_processed], dtype=np.uint8)

        r = preprocess.reward_clip(reward)

        dqn.store(total_step, s_prev, a, r, s, done)

        if total_step > initial_exploration:
            if total_step % update_freq == 0:
                dqn.update(total_step)
            if total_step % target_update_freq == 0:
                dqn.target_update()
            if total_step % save_freq ==0:
                print "saving the model"
                serializers.save_npz('network/{}({}).model'.format(name, comment), dqn.model)   

            epsilon -= epsilon_decrease
            if epsilon < 0.1:
                epsilon = 0.1    

        step += 1
        total_step += 1  
        total_reward += reward
        
        if done:
            f = open("log/{}({}).txt".format(name, comment), "a")
            f.write(str(i_episode+1) + "," + str(total_reward) + ',' + str(step) +"\n")
            f.close()
            print("Episode {} finished after {} timesteps".format(i_episode+1, step))
            print ("total_reward : {}".format(total_reward))
            break
