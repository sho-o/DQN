import numpy as np

class Environment():
	def __init__(self, pics):
		self.walls = self.make_walls()
		self.pics = pics
		self.episode_pics = []

	def make_walls(self, col_size = 3, row_size = 3):
		walls = np.array([[{"up":"y", "down":"y", "right":"y", "left":"y"} for c in range(col_size)] for r in range(row_size)])
		self.make_pink_walls(walls)
		self.make_red_and_green_walls(walls)
		return walls

	def make_pink_walls(self, walls, col_size = 3, row_size = 3):
		for i in range(col_size):
			for j in range(row_size):
				if i == 0: walls[i][j]["up"] = "p"  
				if i == col_size-1: walls[i][j]["down"] = "p"  
				if j == 0: walls[i][j]["left"] = "p"  
				if j == row_size-1: walls[i][j]["right"] = "p"  
		walls[0][0]["right"] = "p"
		walls[0][1]["left"] = "p"

	def make_red_and_green_walls(self, walls, col_size = 3, row_size = 3):
		for key in walls[1][1]:
			walls[1][1][key] = "r"
		walls[1][1]["up"] = "g"
		walls[0][1]["down"]= "g"
		walls[1][0]["right"]= "r"
		walls[2][1]["up"]= "r"
		walls[1][2]["left"]= "r"

	def make_episode_pics(self):
		self.episode_pics = []
		for i in range(10):
			index = np.random.choice(self.pics.shape[0])
			self.episode_pics.append(self.pics[i][index].reshape(1,28,28))

	def s_to_pic(self, s):
		return self.episode_pics[s[0] + s[1]*3 + 1]

	def generate_next_s(self, s, a):
		if self.walls[s[0]][s[1]][a]=="p":
			return s
		else:
			if a == "up": return[s[0]-1, s[1]]
			if a == "down": return[s[0]+1, s[1]]
			if a == "left": return[s[0], s[1]-1]
			if a == "right": return[s[0], s[1]+1]

	def make_reward(self, s, a, reward_clip):
		if self.walls[s[0]][s[1]][a]=="p" or self.walls[s[0]][s[1]][a]=="y":
			return 0
		if self.walls[s[0]][s[1]][a]=="r":
			if reward_clip == True:
				return -1.0
			else:
				return -0.01
		if self.walls[s[0]][s[1]][a]=="g":
			return 1.0

	def judge_finish(self, s):
		return s==[1,1]


if __name__ == '__main__':
	env = Environment("mnist")
	print env.walls
	print env.walls[0][2]["down"]

	print env.generate_next_s([0,0],"left")
	print env.generate_next_s([0,0],"right")
	print env.generate_next_s([0,0],"down")
	print env.generate_next_s([0,0],"up")

	print env.make_reward([0,0],"left")
	print env.make_reward([2,0],"right")
	print env.make_reward([1,2],"left")
	print env.make_reward([0,1],"down")
	print env.make_reward([1,1],"up")

	env.make_episode_pics()
	print len(env.episode_pics)
	print env.episode_pics[9].shape
	print env.s_to_pic([2,2]).shape
