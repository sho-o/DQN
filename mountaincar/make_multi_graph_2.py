import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pylab as plt
import argparse
import pandas as pd
#import glob
import os
import seaborn as sns
import pickle
import re

parser = argparse.ArgumentParser()
parser.add_argument('--folders', '-f', default=[], nargs='+', help='file names')
parser.add_argument('--data_path', '-d', default="log/log.csv", type=str, help='data path')
parser.add_argument('--directory_path', '-dr', default="result2", type=str, help='directory path')
parser.add_argument('--output', '-o', default="output", type=str, help='output names')
parser.add_argument('--x_axis', '-x', default="total_step", type=str, help='x axis')
parser.add_argument('--y_axis', '-y', default="reward", type=str, help='y axis')
parser.add_argument('--rolling_mean', '-r', default=1, type=int, help='use or not rolling mean')
parser.add_argument('--rolling_width', '-w', default=1000, type=int, help='rolling mean width')
parser.add_argument('--add_all_files', '-a', type=bool, default= False, help='add all files in the directory')
parser.add_argument('--key_words', '-k', default=[], nargs='+', help='folder keywords')
parser.add_argument('--mode', '-m', default="training", type=str, help='type of data')
parser.add_argument('--title', '-t', default="", type=str, help='title')
parser.add_argument('--histgram', '-hs', type=bool, default= False, help='make reg_list histgram or not')
parser.add_argument('--update_freq_label', '-l', type=bool, default= False, help='label with update_freq or not')
parser.add_argument('--x_lim', '-xl', default=1000000, type=int, help='x limit')
parser.add_argument('--legend_location', '-ll', default="center right", type=str, help='legend location')
parser.add_argument('--variance', '-v', type=bool, default= False, help='show variance or not')
parser.add_argument('--gradient', '-g', type=bool, default= False, help='show gradient')

args = parser.parse_args()
folders = args.folders
data_path = args.data_path
directory_path = args.directory_path
output = args.output
x_axis = args.x_axis
y_axis = args.y_axis
rolling_mean = args.rolling_mean
rolling_width = args.rolling_width
add_all_files = args.add_all_files
key_words = args.key_words
mode = args.mode
title = args.title
histgram = args.histgram
update_freq_label = args.update_freq_label
x_lim = args.x_lim
legend_location = args.legend_location
variance = args.variance
gradient = args.gradient

sns.set_style("darkgrid")
plt.rcParams['font.size'] = 15

if mode == "test":
	data_path = "evaluation/evaluation.csv"
	y_axis = "reward_mean"
	rolling_mean = 0


def make_graph(folder, c, i):
	ls = "dotted"
	if "f100" in folder:
		ls = "dashed"
	if "f1000" in folder or "qlearning" in folder:
		ls = "solid" 
	df = pd.read_csv("{}/{}/{}".format(directory_path, folder, data_path))
	x = np.array(df.loc[:, x_axis].values)
	y = np.array(df.loc[:, y_axis].values)
	#print x
	#print y
	if rolling_mean == 1:
		y = pd.Series(y).rolling(window=rolling_width).mean()
		#print y
	split = folder.split("_")
	index = None
	for i in range(len(split)):
		if re.match("f" + r"[0-9]+", split[i]):
			index = i
	if update_freq_label == True and index:
		plt.plot(x, y, color = c, label = re.search(r'[0-9]+', split[index]).group(), linestyle = ls)
	else:
		plt.plot(x, y, color = c, label="{}".format(folder), linestyle = ls)
	if mode == "test" and variance == True:
		reward_std = np.array(df.loc[:, "reward_std"].values)
		plt.fill_between(x, y+reward_std, y-reward_std, facecolor=c, alpha=0.3)

def make_ave_value_graph(folder, c, i):
	ls = "dotted"
	if "f100" in folder:
		ls = "dashed"
	if "f1000" in folder or "qlearning" in folder:
		ls = "solid" 
	df = pd.read_csv("{}/{}/log/log.csv".format(directory_path, folder))
	x = np.array(df.loc[:, "total_step"].values)
	y = np.array(df.loc[:, "average_value"].values)
	#print x
	#print y
	#if rolling_mean == 1:
	y = pd.Series(y).rolling(window=100).mean()
		#print y
	#split = folder.split("_")
	#index = None
	#for i in range(len(split)):
		#if re.match("f" + r"[0-9]+", split[i]):
			#index = i
	#if update_freq_label == True and index:
		#plt.plot(x, y, color = c, label = re.search(r'[0-9]+', split[index]).group(), linestyle = ls)
	#else:
	plt.plot(x, y, color = c, label="{}".format(folder), linestyle = ls)

def make_gradient_graph(folder, c, i):
	ls = "dotted"
	if "f100" in folder:
		ls = "dashed"
	if "f1000" in folder or "qlearning" in folder:
		ls = "solid" 
	if os.path.exists("{}/{}/log/abs_grad_list.pickle".format(directory_path, folder)):
		with open("{}/{}/log/abs_grad_list.pickle".format(directory_path, folder)) as f:
			abs_grad_list = pickle.load(f)
		x = np.array(abs_grad_list[0])
		y = np.array(abs_grad_list[1])
		y = pd.Series(y).rolling(window=rolling_width).mean()
		plt.plot(x, y, color = c, label="{}".format(folder), linestyle = ls)

plt.figure(figsize=(8, 6))
colors = ["r", "g", "b", "c", "m", "y", "k", '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
if not os.path.exists("{}/multi_graph".format(directory_path)):
	os.makedirs("{}/multi_graph".format(directory_path))
if add_all_files:
	all_folders = os.listdir('{}'.format(directory_path))
	for f in all_folders:
		if all(key_word in f for key_word in key_words):
			folders.append("{}".format(f))
	folders.sort()
	folders.reverse()
	print folders

for i in range(len(folders)):
	f = folders[i]
	matchObj = re.search("sd", f)
	c_n = int(f[matchObj.end()])
	make_graph(f, colors[c_n], i)
if mode == "training":
	plt.xlabel("training step")
	plt.ylabel("reward(moving average)")
if mode == "test":
	plt.xlabel("training step")
	plt.ylabel("reward mean")
plt.xlim(0, x_lim)
#plt.ylim(-1, 1)
if title == "":
	title = output
	plt.title(title)
else:
	plt.title(title)
plt.legend(loc=legend_location)#, bbox_to_anchor=(0.5,-0.05), ncol=1)
#plt.subplots_adjust(bottom=0.2)
if output == "":
	directory = directory_path.split("/")
	plt.savefig("{}/multi_graph/{}.png".format(directory_path, directory[-1]))
	parent_path = "/".join(directory[:-1])
	plt.savefig("{}/multi_graph/{}.png".format(parent_path, directory[-1]))
else:
	plt.savefig("{}/multi_graph/{}.png".format(directory_path, output))
plt.close()

for i in range(len(folders)):
	f = folders[i]
	matchObj = re.search("sd", f)
	c_n = int(f[matchObj.end()])
	make_ave_value_graph(f, colors[c_n], i)
plt.xlabel("training step")
plt.ylabel("moving average of value")
plt.xlim(0, x_lim)
plt.title("episode average value({})".format(title))
plt.savefig("{}/multi_graph/value_{}.png".format(directory_path, output))
plt.close()

if gradient == True:
	for i in range(len(folders)):
		f = folders[i]
		matchObj = re.search("sd", f)
		c_n = int(f[matchObj.end()])
		make_gradient_graph(f, colors[c_n], i)
	plt.xlabel("training step")
	plt.ylabel("moving average of abs_graient")
	plt.xlim(0, x_lim)
	plt.title("gradient({})".format(title))
	plt.savefig("{}/multi_graph/gradient_{}.png".format(directory_path, output))
	plt.close()
