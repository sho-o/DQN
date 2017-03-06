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