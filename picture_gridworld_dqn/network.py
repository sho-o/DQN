from chainer import Chain
import chainer.functions as F
import chainer.links as L
import numpy as np

class Q(Chain):
	def __init__(self, num_of_actions, n_units=1000):
		super(Q, self).__init__(
			l1=L.Linear(None, n_units),
			l2=L.Linear(None, n_units),
			l3=L.Linear(None, num_of_actions, initialW=np.zeros((num_of_actions, n_units),dtype=np.float32)))

	def __call__(self, x):
		h_1 = F.relu(self.l1(x / 255.0))
		h_2 = F.relu(self.l2(h_1))
		o = self.l3(h_2)
		return o

class Q_conv(Chain):
	def __init__(self, num_of_actions, n_units=1000):
		super(Q_conv, self).__init__(
			conv1=L.Convolution2D(None, 20, 5),
			conv2=L.Convolution2D(None, 50, 5),
			l1=L.Linear(None, 500),
			l2=L.Linear(None, num_of_actions))

	def __call__(self, x):
		h_1 = F.max_pooling_2d(F.relu(self.conv1(x / 255.0)), 2)
		h_2 = F.max_pooling_2d(F.relu(self.conv2(h_1)), 2)
		h_3 = F.relu(self.l1(h_2))
		o = self.l2(h_3)
		return o

class DQN(Chain):
	def __init__(self, num_of_actions):
		super(DQN, self).__init__(
			l1=L.Convolution2D(1, 32, ksize=8, stride=1), #original is stride=4
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