import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matlab.engine
eng = matlab.engine.start_matlab()
import matlab


class ALG:
	def __init__(self,ell,k,lb,ub):
		self.ell = ell
		self.k = k
		self.lb = lb
		self.ub = ub

	def sample(self):
		done = False
		while not done:
			self.delta = self.compute_delta()
			self.x_star = self.get_x_star()
			lb_new = ((self.lb - self.x_star)/(1+(np.sqrt(self.ell)/self.delta))).reshape(-1,1).tolist()
			ub_new = ((self.ub - self.x_star)/(1+(np.sqrt(self.ell)/self.delta))).reshape(-1,1).tolist()
			z = np.ones((int(self.ell),2000)) # shape = (num_groups,num_samples)
			# z = np.asarray(eng.sampling_from_simplex(self.ell,self.k,matlab.double(lb_new),matlab.double(ub_new)))
			ind = np.random.randint(len(z[0]))
			x = self.round(z[:,ind])
			# if self.inN(x):
			# 	done = True
			done = True
		return x

	def compute_delta(self):
		minimum = np.inf
		for j in range(int(self.ell)):
			minimum = min(minimum, 0.5*(self.ub[j]-self.lb[j]) - 1)
		minimum = min(minimum, (self.k - np.sum(self.lb))/self.ell - 1)
		minimum = min(minimum, (np.sum(self.ub) - self.k)/self.ell - 1)
		return minimum

	def get_x_star(self):
		x = np.zeros(int(self.ell))
		x = self.lb + np.ceil(self.delta)
		while np.sum(x) < self.k:
			j = np.argmax(self.ub - np.ceil(self.delta) - x)
			x[j] = min( self.k - np.sum(x) + x[j], self.ub[j] - np.ceil(self.delta))
		return x
	
	def round(self,z):
		y = z + self.x_star
		sorted_indices = np.argsort(y-np.floor(y))[::-1][:len(z)]

		for j in range(int(self.k - np.sum(np.floor(y)))):
			y[sorted_indices[j]] = np.floor(y[sorted_indices[j]]) + 1
		j = int(self.k - np.sum(np.floor(y)))
		while j < self.ell:
			y[sorted_indices[j]] = np.floor(y[sorted_indices[j]]) 
			j += 1
		return y

	def inN(self,x):
		if np.all(np.less_equal(self.lb,x)) and np.all(np.greater_equal(self.ub,x)) and np.sum(x) == self.k:
			return True

	def plot(self):
		plt.scatter(s[0], s[2])
		plt.savefig("test.png")


def main():
	o = ALG(2,10,[0,2],[5,10])
	point = o.sample()
	print(point)

if __name__ == '__main__':
		main()	