class RLP_Log():
	def __init__(self, comment, rlp_iter):
		self.comment = comment
		self.rlp_iter = rlp_iter
		f = open("result/{}/loss_and_penalty/loss_and_penalty.csv".format(comment), "a")
		f.write("episode,total_step,loss,penalty\n")
		f.close()

	def __call__(self, episode, total_step, agt):
		sum_loss = 0
		sum_penalty = 0
		for i in range(self.rlp_iter):
			s, a, r, new_s, done = agt.make_minibatch(total_step)
			loss, penalty = agt.compute_loss(s, a, r, new_s, done, rlp=True)
			sum_loss += loss
			sum_penalty += penalty
		ave_loss = sum_loss/self.rlp_iter
		ave_penalty = sum_penalty/self.rlp_iter
		f = open("result/{}/loss_and_penalty/loss_and_penalty.csv".format(self.comment), "a")
		f.write(str(episode+1) + "," + str(total_step) + "," + str(ave_loss) + "," + str(ave_penalty) + "\n")
		f.close()