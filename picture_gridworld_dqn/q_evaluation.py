import numpy as np
import environment
import copy
import chainer
from chainer import Variable, cuda
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
#need more?

class Q_Evaluation():
	def __init__(self, directory_path, comment, training_pics, actions, q_eval_iter):
		self.directory_path = directory_path
		self.env = environment.Environment(training_pics)
		self.comment = comment
		self.actions = actions
		self.q_eval_iter = q_eval_iter

	def __call__(self, agt, total_step):
		random_state = np.random.get_state()

		all_data = []
		for i in range(self.q_eval_iter):
			self.env.make_episode_pics()
			data = np.array([])
			for s in range(9):
				pic_s = self.env.episode_pics[s]
				q_values = self.Q(pic_s, agt)
				q_values = q_values.reshape(len(self.actions))
				data = np.append(data, q_values)
			all_data.append(data)
		all_data = np.array(all_data)
		mean = all_data.mean(axis=0)
		std =  all_data.std(axis=0)
		self.make_heatmap(mean, total_step, "mean")
		self.make_heatmap(std, total_step, "std")

		np.random.set_state(random_state)

	def Q(self, s, agt):
		if agt.net_type == "full":
			s = np.asarray(s.reshape(1, agt.input_slides*agt.size*agt.size), dtype=np.float32)
		if agt.net_type == "conv" or agt.net_type == "DQN":
			s = np.asarray(s.reshape(1, agt.input_slides, agt.size, agt.size), dtype=np.float32)
		if agt.gpu >= 0:
			s = cuda.to_gpu(s)
		if chainer.__version__ >= "2.0.0":
			s = Variable(s)
		else:
			s = Variable(s, volatile='auto')

		with chainer.no_backprop_mode():
			q = agt.q(s)
			q = q.data[0]
			if agt.gpu >= 0:
				q = cuda.to_cpu(q)
		return q

	def make_heatmap(self, value, total_step, kind):
		valuemap = np.zeros((9,9))
		for s in range(9):
			if s == 4:
				continue
			center = self.calculate_center(s)
			valuemap[center[0], center[1]-1] = value[4*s] #up
			valuemap[center[0], center[1]+1] = value[4*s + 1] #down
			valuemap[center[0]+1, center[1]] = value[4*s + 2] #right
			valuemap[center[0]-1, center[1]] = value[4*s + 3] #left
		sns.heatmap(valuemap, annot=True, cmap='Blues')
		plt.savefig("{}/{}/evaluation/q_evaluation_{}_{}.png".format(self.directory_path, self.comment, total_step, kind))
		plt.close()

	def calculate_center(self, s):
		center0 = (s%3)*3+1
		center1 = (s/3)*3+1
		return [center0, center1]







