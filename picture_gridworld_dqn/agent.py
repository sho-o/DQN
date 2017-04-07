import numpy as np
import copy
from chainer import Variable, optimizers, cuda
import chainer.functions as F
import network

class Agent():
	def __init__(self, exp_policy, net_type, gpu, pic_size, num_of_actions, memory_size, input_slides, batch_size, discount, rms_eps, rms_lr, optimizer_type, mode, threshold, penalty_weight, mix_rate):
		self.exp_policy = exp_policy
		self.net_type = net_type
		self.gpu = gpu
		self.size = pic_size
		self.num_of_actions = num_of_actions
		self.memory_size = memory_size
		self.input_slides = input_slides
		self.batch_size = batch_size
		self.discount = discount
		self.optimizer_type = optimizer_type
		self.epsilon = 1.0
		if self.net_type == "full_connect":
			self.q = network.Q(self.num_of_actions)
		if self.net_type == "convolution":
			self.q = network.Q_conv(self.num_of_actions)
		self.fixed_q = copy.deepcopy(self.q)
		self.replay_memory = {"s":np.zeros((self.memory_size, self.input_slides, self.size, self.size), dtype=np.uint8),
							"a":np.zeros(self.memory_size, dtype=np.uint8),
							"r":np.zeros((self.memory_size), dtype=np.float32),
							"new_s":np.zeros((self.memory_size, self.input_slides, self.size, self.size), dtype=np.uint8),
							"done":np.zeros((self.memory_size), dtype=np.bool)}
		if self.gpu >= 0:
			self.q.to_gpu(self.gpu)
			self.fixed_q.to_gpu(self.gpu)
		if self.optimizer_type == "rmsprop":
			self.optimizer = optimizers.RMSpropGraves(lr=rms_lr, alpha=0.95, momentum=0.95, eps=rms_eps)
		if self.optimizer_type == "sgd":
			self.optimizer = optimizers.SGD(lr=0.01)
		if self.optimizer_type == "adam":
			self.optimizer = optimizers.Adam()
		self.optimizer.setup(self.q)
		self.mode = mode
		self.threshold = threshold
		self.penalty_weight = penalty_weight
		self.mix_rate = mix_rate

	def policy(self, s, eva=False):
		if self.net_type == "full_connect":
			s = np.asarray(s.reshape(1, self.input_slides*self.size*self.size), dtype=np.float32)
		if self.net_type == "convolution":
			s = np.asarray(s.reshape(1, self.input_slides, self.size, self.size), dtype=np.float32)
		if self.gpu >= 0:
			s = cuda.to_gpu(s, device=self.gpu)
		s = Variable(s)

		if self.exp_policy == "epsilon_greedy":
			q = self.q(s)
			q = q.data[0]
			if self.gpu >= 0:
				q = cuda.to_cpu(q)
			q_max = np.amax(q)

			if eva == True:
				epsilon = 0
			else:
				epsilon = self.epsilon

			if np.random.rand() < epsilon:
				action = np.random.randint(0, self.num_of_actions)
			else:
				candidate = np.where(q == q_max)
				action = np.random.choice(candidate[0])
			return action, q[action]

	def store_experience(self, total_step, s, a, r, new_s, done):
		index = total_step % self.memory_size
		self.replay_memory["s"][index] = s
		self.replay_memory["a"][index] = a
		self.replay_memory["r"][index] = r
		self.replay_memory["new_s"][index] = new_s
		self.replay_memory["done"][index] = done

	def q_update(self, total_step):
		s, a, r, new_s, done = self.make_minibatch(total_step)
		self.q.zerograds()
		loss = self.compute_loss(s, a, r, new_s, done)
		loss.backward()
		self.optimizer.update()

	def fixed_q_updqte(self):
		self.fixed_q = copy.deepcopy(self.q)

	def make_minibatch(self, total_step):
		if total_step < self.memory_size:
			index = np.random.randint(0, total_step, self.batch_size)
		else:
			index = np.random.randint(0, self.memory_size, self.batch_size)

		s_batch = np.ndarray(shape=(self.batch_size, self.input_slides, self.size, self.size), dtype=np.float32)
		a_batch = np.ndarray(shape=(self.batch_size, 1), dtype=np.uint8)
		r_batch = np.ndarray(shape=(self.batch_size, 1), dtype=np.float32)
		new_s_batch = np.ndarray(shape=(self.batch_size, self.input_slides, self.size, self.size), dtype=np.float32)
		done_batch = np.ndarray(shape=(self.batch_size, 1), dtype=np.bool)

		for i in range(self.batch_size):
			s_batch[i] = np.asarray(self.replay_memory["s"][index[i]], dtype=np.float32)
			a_batch[i] = self.replay_memory["a"][index[i]]
			r_batch[i] = self.replay_memory["r"][index[i]]
			new_s_batch[i] = np.asarray(self.replay_memory["new_s"][index[i]], dtype=np.float32)
			done_batch[i] = self.replay_memory["done"][index[i]]
		return s_batch, a_batch, r_batch, new_s_batch, done_batch

	def compute_loss(self, s, a, r, new_s, done, rlp=False):
		if self.net_type == "full_connect":
			s = s.reshape(self.batch_size, self.input_slides*self.size*self.size)
			new_s = new_s.reshape(self.batch_size, self.input_slides*self.size*self.size)

		#gpu
		if self.gpu >= 0:
			s = cuda.to_gpu(s, device=self.gpu)
			new_s = cuda.to_gpu(new_s, device=self.gpu)
		s = Variable(s)
		new_s = Variable(new_s)
		q_value = self.q(s)
		q_value_data = q_value.data

		if self.mode == "regularize":
			tg_q_value = self.q(new_s)
		elif self.mode == "target_mix":
			tg_q_value = (1.0-self.mix_rate) * self.q(new_s) + self.mix_rate * self.fixed_q(new_s)
		elif self.model == "default":
			tg_q_value = self.fixed_q(new_s)

		tg_q_value_data = tg_q_value.data

		#cpu
		if self.gpu >= 0:
			q_value_data = cuda.to_cpu(q_value_data)
			tg_q_value_data = cuda.to_cpu(tg_q_value_data)

		max_tg_q_value = np.asarray(np.amax(tg_q_value_data, axis=1), dtype=np.float32)
		target = np.array(q_value_data, dtype=np.float32)
		for i in range(self.batch_size):
			if done[i][0] is True:
				tmp = r[i]
			else:
				tmp = r[i] + self.discount * max_tg_q_value[i]
			target[i, a[i]] = tmp

		#gpu
		if self.gpu >= 0:
			target = cuda.to_gpu(target, device=self.gpu)
		td = Variable(target) - q_value

		#td_clip(keeping variable history)
		td_tmp = td.data + 10.0 * (abs(td.data) <= 1)
		td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)

		zero = np.zeros((self.batch_size, self.num_of_actions), dtype=np.float32)
		if self.gpu >= 0:
			zero = cuda.to_gpu(zero, device=self.gpu)
		zero = Variable(zero)
		loss = F.mean_squared_error(td_clip, zero)

		if self.mode == "regularize" or rlp == True:
			if self.gpu >= 0:
				q_value_data = cuda.to_gpu(q_value_data)
			penalty = F.mean_squared_error(self.fixed_q(s), q_value_data)

			if rlp == True:
				return loss.data, penalty.data

			if penalty.data > self.threshold:
				loss = loss + self.penalty_weight * penalty

		return loss
