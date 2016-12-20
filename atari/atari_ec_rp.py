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
parser.add_argument('--table_size', '-t', type=int, default=10**6, help='table size')
parser.add_argument('--render', '-r', type=int, default=0, help='rendor or not')
parser.add_argument('--epsilon', '-e', type=float, default=0.005)
parser.add_argument('--NearestNeighbor_k', '-k', type=int, default=11)
parser.add_argument('--NearestNeighbor_algo', '-a', type=str, default='kd_tree')
parser.add_argument('--NearestNeighbor_dist', '-d', type=str, default='euclidean')
parser.add_argument('--leaf_size', '-l', type=int, default='1000000')
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
		self.q_table = map(list, list(np.zeros((num_of_actions, table_size))))
		self.s_table = map(list, list(np.random.rand(num_of_actions, table_size, 64)))

	def RP(self, s):
		#RP_start = time.time()
		s_64 = np.dot(s.reshape(1, 84*84), random_matrix)
		#RP_end = time.time() - RP_start
		#print "RP:{}".format(RP_end)
		return s_64

	def NN(self, s, a):
		distance, indices = neigh[a].kneighbors(s) #.reshape(1, 64))
		if distance[0][0] == 0:
			Q = self.q_table[a][indices[0][0]]
			ex_flag = 1
			index = indices[0][0]
		else:
			indices_list = list(indices[0])
			Q_neighbor = [self.q_table[a][i] for i in indices_list]
			Q = sum(Q_neighbor)/len(Q_neighbor)
			ex_flag = 0
			index = 0
		return Q, ex_flag, index

	def Q(self, s):
		ex_flag_all = np.zeros(num_of_actions)
		index_all = np.zeros(num_of_actions, dtype=int)
		q_all = np.zeros(num_of_actions)

		NN_start = time.time()
		for a in range(num_of_actions):
			q_all[a], index_all[a], ex_flag_all[a] = self.NN(s, a)
		NN_end = time.time() - NN_start
		#print "NN:{}".format(NN_end)
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
			q_tmp[data["action"]].append(new_q)
			s_tmp[data["action"]].append(data["s_prev"].reshape(64))
			if hold_matrix[data["action"]][data["index"]] == -1:
				delete_list[data["action"]].append(data["index"])
			else:
				delete_list[data["action"]].append(hold_matrix[data["action"]][data["index"]])
			hold_matrix[data["action"]][data["index"]] = len(q_tmp[data["action"]]) + table_size - 1
		else:
			q_tmp[data["action"]].append(R)
			s_tmp[data["action"]].append(data["s_prev"].reshape(64))
			delete_count[data["action"]] += 1
			hold_matrix[data["action"]][data["index"]] = len(q_tmp[data["action"]]) + table_size - 1

	def delete(self, delete_list):
		count = 0
		for a in range(num_of_actions):
			s2 = time.time()
			#print self.s_table[a].shape
			#print s_table[1].shape
			self.q_table[a].extend(q_tmp[a])
			self.s_table[a].extend(s_tmp[a])
			s3 = time.time()
			for r in delete_list[a]:
				q_table[a].pop(r-count)
				s_table[a].pop(r-count)
				count += 1
			s4 = time.time()
			del self.q_table[a][:delete_count[a]]
			del self.s_table[a][:delete_count[a]]
			s5 = time.time()
			#print s3-s2
			#print s4-s3
			#print s5-s4

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
leaf_size = args.leaf_size
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
	step = 0
	total_reward = 0
	R = 0
	obs = env.reset()
	s = ec_rp.RP(np.zeros((84, 84), dtype=np.uint8))

	start_NN_set = time.time()
	for j in range(num_of_actions):
		neigh[j] = NearestNeighbors(n_neighbors=NN_k, algorithm=NN_algo, metric=NN_dist, leaf_size=leaf_size)
		neigh[j].fit(ec_rp.s_table[j])
	end_NN_set = time.time() - start_NN_set
	print "NN_set{}".format(end_NN_set)

	epi_start = time.time()
	while (True):
		#print "step{}".format(step)
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
		#print "step_time:{}".format(time.time()-step_start)

		if done:
			q_tmp = []
			s_tmp = []
			delete_list = []
			for a in range(num_of_actions):
				q_tmp.append([])
				s_tmp.append([])
				delete_list.append([])

			hold_matrix = -np.ones((num_of_actions, table_size), dtype=int)
			delete_count = [0]*num_of_actions

			print "time_per_step:{}".format((time.time()-epi_start)/step)
			#print "update"
			update_start = time.time()
			for t in range(step):
				R = temporal_memory[step - t - 1]["reward"] + gamma * R
				ec_rp.update(temporal_memory[step - t - 1], R)
			print "update_time:{}".format(time.time()-update_start)

			#print "delete"
			delete_start = time.time()
			ec_rp.delete(delete_list)
			print "delete_time:{}".format(time.time()-delete_start)
			print delete_count
			print delete_list

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