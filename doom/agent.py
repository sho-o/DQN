import numpy as np
import copy
import chainer
from chainer import Variable, optimizers, cuda
import chainer.functions as F
import network

class Agent():
	def __init__(self, exp_policy, net_type, gpu, pic_size, num_of_actions, memory_size, input_slides, batch_size, discount, rms_eps, rms_lr, optimizer_type, mode, threshold, penalty_weight, mix_rate, penalty_function, penalty_type, final_penalty_cut, directory_path, comment):
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
		if self.net_type == "full":
			self.q = network.Q(self.num_of_actions)
		if self.net_type == "conv":
			self.q = network.Q_conv(self.num_of_actions)
		if self.net_type == "DQN":
			self.q = network.DQN(self.num_of_actions)
		self.fixed_q = copy.deepcopy(self.q)
		self.replay_memory = {"s":np.zeros((self.memory_size, self.input_slides, self.size, self.size), dtype=np.uint8),
							"a":np.zeros(self.memory_size, dtype=np.uint8),
							"r":np.zeros((self.memory_size), dtype=np.float32),
							"new_s":np.zeros((self.memory_size, self.input_slides, self.size, self.size), dtype=np.uint8),
							"done":np.zeros((self.memory_size), dtype=np.bool)}
		if self.gpu >= 0:
			self.q.to_gpu()
			self.fixed_q.to_gpu()
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
		self.penalty_function = penalty_function
		self.penalty_type = penalty_type
		self.final_penalty_cut = final_penalty_cut
		self.directory_path = directory_path
		self.comment = comment

	def policy(self, s, eva=False):
		if self.net_type == "full":
			s = np.asarray(s.reshape(1, self.input_slides*self.size*self.size), dtype=np.float32)
		if self.net_type == "conv" or self.net_type == "DQN":
			s = np.asarray(s.reshape(1, self.input_slides, self.size, self.size), dtype=np.float32)
		if self.gpu >= 0:
			s = cuda.to_gpu(s)

		if chainer.__version__ >= "2.0.0":
			s = Variable(s)
		else:
			s = Variable(s, volatile='auto')

		if self.exp_policy == "epsilon_greedy":
			with chainer.no_backprop_mode():
				q = self.q(s)
				q = q.data[0]
				if self.gpu >= 0:
					q = cuda.to_cpu(q)
			q_max = np.amax(q)

			if eva == True:
				epsilon = 0.05
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
		#print a
		self.q.zerograds()
		loss = self.compute_loss(s, a, r, new_s, done)
		with open('{}/{}/gradient/gradient.txt'.format(self.directory_path, self.comment),'a') as f:
			f.write("{},{},{},{},{},{}\n".format(self.q.l5.W.grad[0,1], self.q.l5.W.grad[2,202], self.q.l5.W.grad[3,489], self.q.l4.W.grad[125,2865], self.q.l4.W.grad[398,1629], self.q.l4.W.grad[445,24]))
		loss.backward(retain_grad=True)
		#print "grad", self.q.l5.W.grad[:,0]
		self.optimizer.update()

	def fixed_q_updqte(self):
		self.fixed_q = copy.deepcopy(self.q)

	def make_minibatch(self, total_step):
		if total_step < self.memory_size:
			index = np.random.randint(0, total_step, self.batch_size)
		else:
			index = np.random.randint(0, self.memory_size, self.batch_size)

		s_batch = np.ndarray(shape=(self.batch_size, self.input_slides, self.size, self.size), dtype=np.float32)
		a_batch = np.ndarray(shape=(self.batch_size), dtype=np.int)
		r_batch = np.ndarray(shape=(self.batch_size), dtype=np.float32)
		new_s_batch = np.ndarray(shape=(self.batch_size, self.input_slides, self.size, self.size), dtype=np.float32)
		done_batch = np.ndarray(shape=(self.batch_size), dtype=np.bool)

		for i in range(self.batch_size):
			s_batch[i] = np.asarray(self.replay_memory["s"][index[i]], dtype=np.float32)
			a_batch[i] = self.replay_memory["a"][index[i]]
			r_batch[i] = self.replay_memory["r"][index[i]]
			new_s_batch[i] = np.asarray(self.replay_memory["new_s"][index[i]], dtype=np.float32)
			done_batch[i] = self.replay_memory["done"][index[i]]
		return s_batch, a_batch, r_batch, new_s_batch, done_batch

	def compute_loss(self, s, a, r, new_s, done, loss_log=False):
		if self.net_type == "full":
			s = s.reshape(self.batch_size, self.input_slides*self.size*self.size)
			new_s = new_s.reshape(self.batch_size, self.input_slides*self.size*self.size)

		#gpu
		if self.gpu >= 0:
			s = cuda.to_gpu(s)
			new_s = cuda.to_gpu(new_s)
		if chainer.__version__ >= "2.0.0":
			s = Variable(s)
			new_s = Variable(new_s)
		else:
			s = Variable(s, volatile='auto')
			new_s = Variable(new_s, volatile='auto')
		q_value = self.q(s)

		with chainer.no_backprop_mode():
			if self.mode == "regularize":
				tg_q_value = self.q(new_s)
			elif self.mode == "target_mix":
				tg_q_value = (1.0-self.mix_rate) * self.q(new_s) + self.mix_rate * self.fixed_q(new_s)
			elif self.mode == "default":
				tg_q_value = self.fixed_q(new_s)
		#print "tg_q_value[0]", tg_q_value[0].data

		if self.gpu >= 0:
			a = cuda.to_gpu(a)
			r = cuda.to_gpu(r)
			done = cuda.to_gpu(done)

		if chainer.__version__ >= "2.0.0":
			a = Variable(a)
		else:
			a = Variable(a, volatile='auto')

		argmax_a = F.argmax(tg_q_value, axis=1)

		#print a
		#print r
		q_action_value = F.select_item(q_value, a)
		#print "q_action_value", q_action_value.data
		target = r + self.discount * (1.0 - done) * F.select_item(tg_q_value, argmax_a)
		#print "target", target.data
		#target is float32

		q_action_value = F.reshape(q_action_value, (-1, 1))
		target = F.reshape(target, (-1, 1))

		loss_sum = F.sum(F.huber_loss(q_action_value, target, delta=1.0))
		loss = loss_sum / q_action_value.shape[0]
		#print "loss_a", loss.data

		if self.mode == "regularize" or loss_log == True:
			if self.penalty_function == "value":
				y = q_value
				with chainer.no_backprop_mode():
					t = self.fixed_q(s)
			if self.penalty_function == "action_value":
				y = q_action_value
				with chainer.no_backprop_mode():
					t = F.select_item(self.fixed_q(s), a)
					t = F.reshape(t, (-1, 1))
			if self.penalty_function == "max_action_value":
				y = F.select_item(self.q(new_s), argmax_a)
				y = F.reshape(y, (-1, 1))
				with chainer.no_backprop_mode():
					t = F.select_item(self.fixed_q(new_s), argmax_a)
					t = F.reshape(t, (-1, 1))

			if self.penalty_type == "huber":
				if self.final_penalty_cut == 1:
					penalty_sum = F.sum((1.0 - done)*F.huber_loss(y, t, delta=1.0))
				else:
					penalty_sum = F.sum(F.huber_loss(y, t, delta=1.0))
				penalty = penalty_sum / (y.shape[0]*y.shape[1])
			if self.penalty_type == "mean_squared":
				penalty = F.mean_squared_error(y, t)

			if loss_log == True:
				#y_data = cuda.to_cpu(y.data)
				#t_data = cuda.to_cpu(t.data)
				return loss, penalty
				#return loss, penalty, np.average(y_data), np.std(y_data), np.average(t_data), np.std(t_data)

			if penalty.data > self.threshold:
				#print "-------------on----------------"
				loss = loss + self.penalty_weight * penalty
		#print "loss_b", loss.data
		return loss
