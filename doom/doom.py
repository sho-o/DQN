#coding:utf-8
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
import time
import ppaquette_gym_doom

parser = argparse.ArgumentParser()
parser.add_argument('--name', '-N', default='DoomDefendCenter-v0', type=str, help='game name')
parser.add_argument('--comment', '-c', default='', type=str, help='comment to distinguish output')                    
parser.add_argument('--gpu', '-g', default= -1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--randomskip', '-rs', default=1, type=int, help='randomskip the frames or not')
parser.add_argument('--n_episode', '-ne', default=10**5, type=int, help='number of episode to learn')
parser.add_argument('--n_step', '-n', default=5*10**7, type=int, help='number of steps to learn')
parser.add_argument('--actionskip', '-as', default=4, type=int, help='number of action repeating')
parser.add_argument('--update', '-u', default=4, type=int, help='update freaquency')
parser.add_argument('--targetupdate', '-t', default=10**4, type=int, help='target update freaquency')
parser.add_argument('--save_freq', '-sf', default=5*10**4, type=int, help='evaluation frequency')
parser.add_argument('--eval_freq', '-ef', default=10*10**4, type=int, help='evaluation frequency')
parser.add_argument('--eval_step', '-es', default=10*10**4, type=int, help='evaluation steps')
parser.add_argument('--eval_epsilon', '-ee', default=0.05, type=float, help='evaluation epsilon')
parser.add_argument('--eval_switch', '-e', default=0, type=int, help='evaluation switch')
parser.add_argument('--initial', '-i', default=5*10**4, type=int, help='number of initial exploration')
parser.add_argument('--epsilon_end', '-ep', default=10**6, type=float, help='the step number of the end of epsilon decrease')
parser.add_argument('--batchsize', '-b', type=int, default=32, help='learning minibatch size')
parser.add_argument('--memorysize', '-m', type=int, default=10**6, help='replay memory size')
parser.add_argument('--inputslides', '-sl', type=int, default=4, help='number of input slides')
parser.add_argument('--render', '-r', type=int, default=0, help='rendor or not')
parser.add_argument('--initial_network', '-in', type=int, default=0, help='use initial_network or not')
parser.add_argument('--load', '-l', type=str, default='DoomDefendCenter-v0_.model', help='initial network')
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

        if self.gpu >= 0:
            self.model.to_gpu(gpu)
            self.target_model.to_gpu(gpu)

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

        if self.gpu >= 0:
            s_prev_replay = cuda.to_gpu(s_prev_replay, device=gpu)
            s_replay = cuda.to_gpu(s_replay, device=gpu)

        self.model.zerograds()
        loss = self.compute_loss(s_prev_replay, a_replay, r_replay, s_replay, done_replay)
        #print loss.data
        loss.backward()
        self.optimizer.update()
        #print "W{}".format(self.model.l5.W.data[1][5])
  

    def compute_loss(self, s_prev, a, r, s, done):
        s_prev = Variable(s_prev)
        s = Variable(s)

        q = self.model(s_prev)
        q_data = q.data
        if self.gpu >= 0:
            q_data = cuda.to_cpu(q_data)

        q_target = self.target_model(s)
        q_target_data = q_target.data
        if self.gpu >= 0:
            q_target_data = cuda.to_cpu(q_target_data)
        max_q_target = np.asarray(np.amax(q_target_data, axis=1), dtype=np.float32)

        target = np.asanyarray(copy.deepcopy(q_data), dtype=np.float32)
        
        for i in range(self.batch_size):
            if done[i][0] is True:
                tmp = r[i]
            else:
                tmp = r[i] + self.gamma * max_q_target[i]
            target[i, a[i]] = tmp

        if self.gpu >= 0:
            target = cuda.to_gpu(target, device=gpu)
        td = Variable(target) - q
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)

        zero = np.zeros((self.batch_size, self.num_of_actions), dtype=np.float32)
        if self.gpu >= 0:
            zero = cuda.to_gpu(zero, device=gpu)
        zero = Variable(zero)    
        loss = F.mean_squared_error(td_clip, zero)
        return loss

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
            candidate = np.where(q == np.amax(q))
            action = np.random.choice(candidate[0])   
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
update_freq = args.update
target_update_freq = args.targetupdate
save_freq = args.save_freq
eval_freq = args.eval_freq
eval_step = args.eval_step
eval_epsilon = args.eval_epsilon
eval_switch = args.eval_switch
initial_exploration = args.initial
memory_size = args.memorysize
input_slides = args.inputslides
batch_size = args.batchsize
render = args.render
n_step = args.n_step
ini_net = args.initial_network
load = args.load
epsilon_decrease = 0.9/(args.epsilon_end-args.initial)
epsilon = 1.0
total_step = 0
update_times = 0
target_update_times = 0
eval_counter = 0
#max_average_reward = -10*6
num_of_actions = 4
if ini_net == 1:
    epsilon = 0.1
if gpu >= 0:
    cuda.get_device(gpu).use()

def evaluation():
    global env_eval
    global eval_counter
    #global max_average_reward

    total_step = 0
    accum_reward = 0
    eval_counter += 1
    for eval_episode in range(n_episode):
        obs = env_eval.reset()
        
        s = np.zeros((4, 84, 84), dtype=np.uint8)
        s[3] = preprocess.one(obs)

        while (True):
            if render == 1:
                env_eval.render()

            #ale
            if randomskip == 0:
                reward = 0
                a = dqn.epsilon_greedy(s, eval_epsilon)
                action = dqn.action_convert(a)
                action = env_eval._action_set[action]
                for i in range(action_skip):
                    reward += env_eval.ale.act(action)
                    obs_prev = copy.deepcopy(obs)
                    obs = env_eval._get_obs()
                done = env_eval.ale.game_over()
                obs_processed = preprocess.two(obs_prev, obs)

            #gym
            if randomskip == 1:
                a = dqn.epsilon_greedy(s, epsilon)
                action = dqn.action_convert(a)
                obs, reward, done, info = env_eval.step(action)
                obs_processed = preprocess.one(obs)

            s_prev = copy.deepcopy(s)
            s = np.asanyarray([s[1], s[2], s[3], obs_processed], dtype=np.uint8)

            total_step += 1 
            accum_reward += reward
            
            if done:
                break

        if total_step > eval_step:
            average = accum_reward/float(eval_episode+1.0)
            f = open("evaluation/{}_{}.txt".format(name, comment), "a")
            f.write(str(eval_counter) + "," + str(average) + "," + str(time.time()-start) + "\n")
            f.close()
            #if average > max_average_reward:
                #max_average_reward = copy.deepcopy(average)
                #print "Evaluation {}: max average reward is {}".format(eval_counter, max_average_reward)
                #print "-------------------------saving the model-------------------------------"
                #serializers.save_npz('network/{}_{}.model'.format(name, comment), dqn.model)
            #break

#main
env = gym.make('ppaquette/{}'.format(name))
env_eval = gym.make('ppaquette/{}'.format(name))
dqn = DQN(gpu, num_of_actions, memory_size, input_slides, batch_size)
if ini_net == 1:
    print "-----------------use {} as initial network-------------------".format(load)
    serializers.load_npz('network/{}'.format(load), dqn.model)
    serializers.load_npz('network/{}'.format(load), dqn.target_model)
preprocess = Preprocess()
game_start = time.time()

for i_episode in range(n_episode):
    episode_start = time.time()
    step = 0
    total_reward = 0
    obs = env.reset()
    
    s = np.zeros((4, 84, 84), dtype=np.uint8)
    s[3] = preprocess.one(obs)

    while (True):
        if render == 1:
            env.render()
            
        #gym
        if randomskip == 1:
            a = dqn.epsilon_greedy(s, epsilon)
            action = dqn.action_convert(a)
            obs, reward, done, info = env.step(action)
            obs_processed = preprocess.one(obs)    

        s_prev = copy.deepcopy(s)
        s = np.asanyarray([s[1], s[2], s[3], obs_processed], dtype=np.uint8)
        #plt.gray()
        #plt.imshow(s[3])
        #plt.pause(0.0001)
        #plt.clf()

        r = preprocess.reward_clip(reward)

        dqn.store(total_step, s_prev, a, r, s, done)

        step += 1
        total_step += 1  
        total_reward += reward

        #Learning
        if total_step > initial_exploration:
            if total_step % update_freq == 0:
                dqn.update(total_step)
                update_times += 1
             
            if total_step % target_update_freq == 0:
            	print "---------------------------target update--------------------------------"
                dqn.target_update()
                target_update_times += 1

            epsilon -= epsilon_decrease
            if epsilon < 0.1:
                epsilon = 0.1   

            if total_step % eval_freq == 0 and eval_switch == 1:
                print "-----------------------------Evaluation--------------------------------"
                evaluation()

            if total_step % save_freq == 0:
                print "-------------------------saving the model-------------------------------"
                serializers.save_npz('network/{}_{}.model'.format(name, comment), dqn.model)    

        if done:
            episode_time = time.time() - episode_start
            total_time = time.time()-game_start
            f = open("log/{}_{}.txt".format(name, comment), "a")
            f.write(str(i_episode+1) + "," + str(total_reward) + ',' + str(step) + ',' + str(total_step) + ',' + str(total_time) + "\n")
            f.close()
            print("-------------------Episode {} finished after {} steps-------------------".format(i_episode+1, step))
            print ("total_reward : {}".format(total_reward))
            print ("total_step : {}".format(total_step))
            print ("epsilon : {}".format(epsilon))
            print ("update_times : {}".format(update_times))
            print ("target_update_times: {}".format(target_update_times))
            print ("total_time: {}".format(total_time))
            print ("episode_time: {}".format(episode_time))
            print ("time/step: {}".format(episode_time/step))
            break

    if total_step > n_step:
        break    