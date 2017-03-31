#import numpy as np
import environment
from chainer import serializers
import argparse
#from matplotlib import pylab as plt
import network

parser = argparse.ArgumentParser()
parser.add_argument('--load_nets', '-l', default=[], nargs='+', help='load nets')
parser.add_argument('--pic_kind', '-k', default="mnist", type=str, help='kind of pictures')
args = parser.parse_args()

num_of_actions = 4
load_nets = args.load_nets
pic_kind = args.pic_kind

env = environment.Environment(pic_kind)
env.make_episode_pics()
print "[up, down, right, left]"

for load in load_nets:
	if "full" in load:
		q = network.Q(num_of_actions)
	if "conv" in load:
		q = network.Q_conv(num_of_actions)
	print load

	serializers.load_npz('{}/network/q.net'.format(load), q)
	for i in range (1,10):
		s = [(i-1)%3, (i-1)/3]
		pic_s = env.s_to_pic(s)
		print i, q(pic_s).data




