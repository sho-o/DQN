from chainer import Chain
import chainer.functions as F
import chainer.links as L
import numpy as np

class DQN(Chain):
    def __init__(self, num_of_actions):
        super(DQN, self).__init__(
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