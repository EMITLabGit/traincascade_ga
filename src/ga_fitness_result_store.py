"""
Class for storing the fitness values across generations.
"""

import numpy as np


# class to store and update fitness values of individuals across generations
class FitnessStore:

    def __init__(self, ga_instance):
        self.ga_instance = ga_instance
        self.feature_fitness_dict = {}

    def _update_fitness_score(self, individual, fitness):
        if individual in self.feature_fitness_dict:
            self.feature_fitness_dict[individual].append(fitness)
        else:
            self.feature_fitness_dict[individual] = [fitness]

    def update_fitness_scores(self, solutions, val_auc,
                              feature_importance_dict):
        # for each individual in the population, update the fitness score
        # there can be duplicate features (individuals) in the population

        # group the importance values of the duplicate features and take the sum
        # use a temporary dictionary to store the importance values of the duplicate features
        temp_dict = {}

        for idx in range(len(solutions)):
            individual = tuple(solutions[idx])
            feat_key = f"f{idx}"
            importance = feature_importance_dict.get(feat_key, 0)
            if individual in temp_dict:
                temp_dict[individual] += importance
            else:
                temp_dict[individual] = importance

        total_gain = sum(temp_dict.values())

        for individual, importance in temp_dict.items():
            fitness = (val_auc - 0.5) * importance / total_gain
            self._update_fitness_score(individual, fitness)

    def get_fitness_scores(self, solutions):
        # get the average fitness score of each individual across generations
        fitness_scores = []

        for idx in range(len(solutions)):
            individual = tuple(solutions[idx])
            fitness_scores.append(
                np.mean(self.feature_fitness_dict[individual]))

        # if all elements in fitness_scores is zero, then set it to 1
        if np.sum(fitness_scores) == 0:
            fitness_scores = [1 for _ in range(len(fitness_scores))]

        return fitness_scores

    def get_avg_score(self, top_k):
        # get the average fitness score of the top k percentile individuals across all generations

        fitness_scores = []
        for individual, scores in self.feature_fitness_dict.items():
            fitness_scores.append(np.mean(scores))

        fitness_scores = sorted(fitness_scores, reverse=True)

        return np.mean(fitness_scores[:top_k])

    def get_best_features(self, K):
        # get the top K features across all generations based on the average fitness score

        fitness_scores, features = [], []

        for individual, scores in self.feature_fitness_dict.items():
            fitness_scores.append(np.mean(scores))
            features.append(individual)

        fitness_scores, features = zip(
            *sorted(zip(fitness_scores, features), reverse=True))

        return np.array(features[:K], dtype=np.int32)
