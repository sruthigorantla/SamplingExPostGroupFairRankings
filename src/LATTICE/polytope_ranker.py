import numpy as np
from numpy.random import default_rng
import pandas as pd
import argparse

import copy 
import math

from itertools import product
from collections import defaultdict, Counter

from LATTICE.lattice_point_sampler import LatticePointSampler



class LatticeSampler(object):
	"""docstring for ClassName"""

	def __init__(self, true_ranking, true_scores, id_2_group, num_groups, proportions, p_deviation, flag, k):
		"""initialize the class and its corresponding DP table"""

		self.true_ranking = true_ranking
		self.true_scores = true_scores
		self.id_2_group = id_2_group

		self.rank_len = k
		self.num_groups = num_groups
		self.proportions = proportions
		self.flag = flag
		self.intra_group_ranking = self.get_intra_group_ranking()
		

		self.LB, self.UB = self.get_bounds(p_deviation)

		self.sampler = LatticePointSampler(self.num_groups, self.rank_len, self.LB, self.UB)




	def get_intra_group_ranking(self):

		intra_group_ranking = defaultdict(list)

		counter = np.zeros(self.num_groups)
		for item in self.true_ranking:
			intra_group_ranking[self.id_2_group[item]].append(item)
			counter[self.id_2_group[item]] += 1
			if (counter > self.rank_len).all():
				break

		return intra_group_ranking


	def get_bounds(self, delta):

		# Generating L_k and U_k vectors
		L_k = []
		U_k = []
		for j in range(self.num_groups):
			L_k.append(math.ceil((self.proportions[j]-delta)*self.rank_len))
			U_k.append(math.floor((self.proportions[j]+delta)*self.rank_len))

		return L_k, U_k


	def construct_ranking(self, answer):

		final_ranking = []

		intra_group_ranking = copy.deepcopy(self.intra_group_ranking)

		for item in answer:
			final_ranking.append(intra_group_ranking[item].pop(0))

		return final_ranking


	def sample_ranking(self, num_samples):

		all_sampled_points = np.array(self.sampler.sample(num_samples), dtype=int)
		# print(all_sampled_points)
		# exit()

		assert len(all_sampled_points) == num_samples


		final_rankings = []
		for sampled_point in all_sampled_points:
			assert sum(sampled_point) == self.rank_len, f"Total sum of sampled point is {sum(sampled_point)} instead of {self.rank_len}"
			
			# Get a random permutation of groups for the ranking

			permutation = []
			for group in range(self.num_groups):
				permutation += [group]*sampled_point[group]

			np.random.shuffle(permutation)


			# Need to map the group-wise ranking to  document ids
			final_rankings.append(self.construct_ranking(permutation))

		return final_rankings
		


