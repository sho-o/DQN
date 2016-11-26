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
from sklearn.neighbors import NearestNeighbors

parser = argparse.ArgumentParser()
parser.add_argument('--name', '-N', default='Pong-v0', type=str, help='game name')
parser.add_argument('--comment', '-c', default='', type=str, help='comment to distinguish output')                    
parser.add_argument('--gpu', '-g', default= -1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--randomskip', '-rs', default=0, type=int, help='randomskip the frames or not')
parser.add_argument('--n_episode', '-ne', default=10**5, type=int, help='number of episode to learn')
parser.add_argument('--n_step', '-n', default=5*10**7, type=int, help='number of steps to learn')
parser.add_argument('--actionskip', '-as', default=4, type=int, help='number of action repeating')
parser.add_argument('--table_size', '-m', type=int, default=10**6, help='table size')
parser.add_argument('--render', '-r', type=int, default=0, help='rendor or not')
parser.add_argument('--epsilon', '-e', type=float, default=0.005)
parser.add_argument('--NearestNeighbor_k', '-k', type=int, default=11)
parser.add_argument('--NearestNeighbor_algo', '-a', type=str, default='brute')
parser.add_argument('--NearestNeighbor_dist', '-d', type=str, default='euclidean')
parser.add_argument('--gamma', '-gam', type=float, default=1)

args = parser.parse_args()
class Preprocess():
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

class EC_RP():
	def __init__(self):
		self.q_table = np.zeros((num_of_actions, table_size))
		self.s_table = np.zeros((num_of_actions, table_size, 64))

	def RP(self, s):
		s_64 = np.dot(s.reshape(1, 84*84), random_matrix)
		return s_64

	def NN(self, s, a):
		indices = neigh[a].kneighbors(s, return_distance=False) #.reshape(1, 64))
		Q_neighbor = self.q_table[a][indices]
		Q_ave = np.average(Q_neighbor)
		return Q_ave

	def search(self, s, a):
		dif = self.s_table[a] - s
		match = dif.any(axis=1)
		indices = np.where(match == False)
		if indices[0].size == 0:
			ex_flag = 0
			index = 0
		else:
			ex_flag = 1
			index = indices[0][0]
		return ex_flag, index

	def Q(self, s):
		q_all = np.zeros(num_of_actions)
		ex_flag_all = np.zeros(num_of_actions)
		index_all = np.zeros(num_of_actions)
		for a in range(num_of_actions):
			ex_flag, index = self.search(s, a)
			if ex_flag == 1:
				q_all[a] = self.q_table[a][index]
			else:
				q_all[a] = self.NN(s, a)
			ex_flag_all[a] = ex_flag
			index_all[a] = index
		return q_all, ex_flag_all, index_all

	def epsilon_greedy(self, s, epsilon):
		q, ex_flag, index = self.Q(s)
		if np.random.rand() < epsilon:
			action = np.random.randint(0, num_of_actions)
		else:
			candidate = np.where(q == np.amax(q))
			action = np.random.choice(candidate[0])
		return action, ex_flag[action], index[action]

	def update(self, data, R):
		if data["ex_flag"] == 1:
			new_q = max(R, self.q_table[data["action"]][data["index"]])
			self.q_table[data["action"]][data["index"]] = new_q
			q_tmp[data["action"]] = np.append(q_tmp[data["action"]], new_q)
			s_tmp[data["action"]] = np.append(s_tmp[data["action"]], data["s_prev"].reshape(1,64), axis=0)
			delete_list[data["action"]].append(data["index"])
		else:
			q_tmp[data["action"]] = np.append(q_tmp[data["action"]], R)
			s_tmp[data["action"]] = np.append(s_tmp[data["action"]], data["s_prev"].reshape(1,64), axis=0)
			delete_count[data["action"]] += 1

	def delete(self, delete_list, delete_count):
		for a in range(num_of_actions):
			#print q_tmp[a].shape
			#print s_tmp[a].shape
			#print len(delete_list[a])
			#print delete_count[a]
			q_tmp[a] = np.delete(q_tmp[a], delete_list[a], axis=0)
			s_tmp[a] = np.delete(s_tmp[a], delete_list[a], axis=0)
			q_tmp[a] = np.delete(q_tmp[a], np.s_[0:(q_tmp[a].shape[0]-table_size)], axis=0)
			s_tmp[a] = np.delete(s_tmp[a], np.s_[0:(s_tmp[a].shape[0]-table_size)], axis=0)
			#print q_tmp[a].shape
			#print s_tmp[a].shape
			self.q_table[a] = q_tmp[a]
			self.s_table[a] = s_tmp[a]

gpu = args.gpu
name = args.name
comment = args.comment
randomskip = args.randomskip
n_episode = args.n_episode
action_skip = args.actionskip
table_size = args.table_size
render = args.render
n_step = args.n_step
epsilon = args.epsilon
gamma = args.gamma
total_step = 0
NN_k = args.NearestNeighbor_k
NN_algo = args.NearestNeighbor_algo
NN_dist = args.NearestNeighbor_dist
if gpu >= 0:
	cuda.get_device(gpu).use()
random_matrix = np.random.randn(84*84, 64)
#main
env = gym.make(name)
num_of_actions = env.action_space.n
preprocess = Preprocess()
ec_rp = EC_RP()
start = time.time()

for i_episode in range(n_episode):
	neigh = [[]]*num_of_actions
	temporal_memory = []
	delete_list = [[]]*num_of_actions
	delete_count = [0]*num_of_actions
	step = 0
	total_reward = 0
	R = 0
	obs = env.reset()
	s = ec_rp.RP(np.zeros((84, 84), dtype=np.uint8))
	print "start"
	for j in range(num_of_actions):
		neigh[j] = NearestNeighbors(n_neighbors=NN_k, algorithm=NN_algo, metric=NN_dist)
		neigh[j].fit(ec_rp.s_table[j])
		print "end{}".format(j)

	while (True):
		if render == 1:
			env.render()
		#ale
		if randomskip == 0:
			reward = 0
			a, ex_flag, index = ec_rp.epsilon_greedy(s, epsilon)
			action = env._action_set[a]
			for i in range(action_skip):
				reward += env.ale.act(action)
				obs_prev = copy.deepcopy(obs)
				obs = env._get_obs()
				done = env.ale.game_over()
				obs_processed = preprocess.two(obs_prev, obs)

		#gym
		if randomskip == 1:
			a, ex_flag, index = ec_rp.epsilon_greedy(s, epsilon)
			obs, reward, done, info = env.step(a)
			obs_processed = preprocess.one(obs)

		s_prev = copy.deepcopy(s)
		s = ec_rp.RP(np.asanyarray(obs_processed, dtype=np.uint8))
		#plt.gray()
		#plt.imshow(obs_processed)
		#plt.pause(0.0001)
		#plt.clf()

		#r = preprocess.reward_clip(reward)
		data = {"action":int(a), "s_prev":s_prev, "reward":reward, "ex_flag":ex_flag, "index":int(index)}
		temporal_memory.append(data)

		step += 1
		total_step += 1
		total_reward += reward
		print "step{}".format(step)

		if done:
			q_tmp = list(ec_rp.q_table)
			s_tmp = list(ec_rp.s_table)
			print "update"
			for t in range(step):
				R = temporal_memory[step - t - 1]["reward"] + gamma * R
				ec_rp.update(temporal_memory[step - t - 1], R)
			print "delete"
			ec_rp.delete(delete_list, delete_count)

			total_time = time.time()-start
			f = open("log/{}_{}.txt".format(name, comment), "a")
			f.write(str(i_episode+1) + "," + str(total_reward) + ',' + str(step) + ',' + str(total_step) + ',' + str(total_time) + "\n")
			f.close()
			print("-------------------Episode {} finished after {} steps-------------------".format(i_episode+1, step))
			print ("total_reward : {}".format(total_reward))
			print ("total_step : {}".format(total_step))
			print ("total_time: {}".format(total_time))
			break

		if total_step > n_step:
			break