import numpy as np
from numpy.random import default_rng
import pandas as pd
import argparse

import copy 
import math

from itertools import product
from collections import defaultdict, Counter

class FairEpsSampler:

	def __init__(self, true_ranking, true_scores, id_2_group, num_groups, proportions, p_deviation, flag, k, eps):

		self.true_ranking = true_ranking
		self.true_scores = true_scores
		self.id_2_group = id_2_group
		self.delta = p_deviation
		
		self.rank_len = k
		self.num_groups = num_groups
		self.proportions = proportions
		self.eps = eps
		self.flag = flag
		self.intra_group_ranking = self.get_intra_group_ranking()


	def get_intra_group_ranking(self):

		intra_group_ranking = defaultdict(list)

		counter = np.zeros(self.num_groups)
		for item in self.true_ranking:
			intra_group_ranking[self.id_2_group[item]].append(item)
			counter[self.id_2_group[item]] += 1
			if (counter > self.rank_len).all():
				break

		return intra_group_ranking



	def sample_algo(self):

		counts = np.zeros(self.num_groups)
		group_assignment = []

		for i in range(self.rank_len):
			
			found = False
			r = np.random.uniform(low=0.0, high=1.0, size=None)
			
			if r > self.eps:
				for group in range(self.num_groups):

					if counts[group] < (math.ceil((self.proportions[group]-self.delta)*(i+1))):
						found = True
						group_assignment.append(group)
						counts[group] += 1
						break
				
			if not found:
				group = np.random.randint(self.num_groups)
				group_assignment.append(group)
				counts[group] += 1

		return group_assignment



	def construct_ranking(self, answer):

		final_ranking = []
		intra_group_ranking = copy.deepcopy(self.intra_group_ranking)
		for item in answer:
			final_ranking.append(intra_group_ranking[item].pop(0))
		return final_ranking


	def sample_ranking(self, num_samples):


		final_rankings = []
		for rank_id in range(num_samples):

			sampled_rank = self.sample_algo()
			assert len(sampled_rank) == self.rank_len, f"Total sum of sampled point is {sum(sampled_point)} instead of {self.rank_len}"
			
			# Need to map the group-wise ranking to  document ids
			final_rankings.append(self.construct_ranking(sampled_rank))

		return final_rankings


		
