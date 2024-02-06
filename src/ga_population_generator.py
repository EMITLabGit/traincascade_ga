"""
Code to generate initial population for GA that satify the constraints with which haar features are generated
"""

import numpy as np
import pandas as pd


class HaarFeatureSampler:

    def __init__(self):
        self.count_haar_features_dict = {
            1: 43200,
            2: 43200,
            3: 27600,
            4: 27600,
            5: 20736
        }

        # dispatcher dictionary for calling the appropriate feature sampling function
        self.feat_sample_funcs = {
            1: self._sample_feature_1,
            2: self._sample_feature_2,
            3: self._sample_feature_3,
            4: self._sample_feature_4,
            5: self._sample_feature_5
        }

        # x, w combinations
        self.combinations_x_w = np.array([(x, w) for w in range(1, 25)
                                          for x in range(24 - w + 1)])
        # x, hw combinations
        self.combinations_x_hw = np.array([(x, hw) for hw in range(1, 13)
                                           for x in range(24 - 2 * hw + 1)])
        # x, tw combinations
        self.combinations_x_tw = np.array([(x, tw) for tw in range(1, 9)
                                           for x in range(24 - 3 * tw + 1)])

        self.feature_type_probabilities = np.array(
            [self.count_haar_features_dict[i] for i in range(1, 6)])
        self.feature_type_probabilities = self.feature_type_probabilities / np.sum(
            self.feature_type_probabilities)

    def sample_feature_type(self, n_samples):
        return np.random.choice(np.array(
            list(self.count_haar_features_dict.keys())),
                                size=n_samples,
                                p=self.feature_type_probabilities)

    def sample(self, n_samples):
        # first choose the feature type of the n_population features in proportion to the number of features of each type
        # probability of choosing a feature type is proportional to the number of features of that type
        feature_types = self.sample_feature_type(n_samples)

        # once the feature type is chosen, generate the feature by uniformly sampling from the range of possible values
        features = np.array(
            [self._sample_feature(ft_type) for ft_type in feature_types])

        # features[-30:] = features[:30]

        return features

    def _sample_feature(self, ft_type):
        try:
            sampled_feature = self.feat_sample_funcs[ft_type]()
        except KeyError as e:
            raise ValueError(f"Invalid feature type {ft_type}")
        return sampled_feature

    def _sample_feature_1(self):
        # (x, w) are sampled first and then (y, h) are sampled since (x, w) are independent of (y, h)

        # (x, hw) are sampled first from self.combinations_x_hw and converted to (x, w) by w = 2 * hw
        x, hw = self.combinations_x_hw[np.random.randint(
            0, len(self.combinations_x_hw))]
        w = 2 * hw

        # (y, h) are sampled from self.combinations_x_w
        y, h = self.combinations_x_w[np.random.randint(
            0, len(self.combinations_x_w))]

        return np.array([1, x, y, w, h])

    def _sample_feature_2(self):
        # (x, w) are sampled first and then (y, h) are sampled since (x, w) are independent of (y, h)

        # (x, w) are sample from self.combinations_x_w
        x, w = self.combinations_x_w[np.random.randint(
            0, len(self.combinations_x_w))]

        # (y, hh) are sampled first from self.combinations_x_hw and converted to (y, h) by h = 2 * hh
        y, hh = self.combinations_x_hw[np.random.randint(
            0, len(self.combinations_x_hw))]
        h = 2 * hh

        return np.array([2, x, y, w, h])

    def _sample_feature_3(self):
        # (x, w) are sampled first and then (y, h) are sampled since (x, w) are independent of (y, h)

        # (x, tw) are sampled first from self.combinations_x_tw and converted to (x, w) by w = 3 * tw
        x, tw = self.combinations_x_tw[np.random.randint(
            0, len(self.combinations_x_tw))]
        w = 3 * tw

        # (y, h) are sampled from self.combinations_x_w
        y, h = self.combinations_x_w[np.random.randint(
            0, len(self.combinations_x_w))]

        return np.array([3, x, y, w, h])

    def _sample_feature_4(self):
        # (x, w) are sampled first and then (y, h) are sampled since (x, w) are independent of (y, h)

        # (x, w) are sample from self.combinations_x_w
        x, w = self.combinations_x_w[np.random.randint(
            0, len(self.combinations_x_w))]

        # (y, th) are sampled first from self.combinations_x_tw and converted to (y, h) by h = 3 * th
        y, th = self.combinations_x_tw[np.random.randint(
            0, len(self.combinations_x_tw))]
        h = 3 * th

        return np.array([4, x, y, w, h])

    def _sample_feature_5(self):
        # (x, w) are sampled first and then (y, h) are sampled since (x, w) are independent of (y, h)

        # (x, hw) are sampled first from self.combinations_x_hw and converted to (x, w) by w = 2 * hw
        x, hw = self.combinations_x_hw[np.random.randint(
            0, len(self.combinations_x_hw))]
        w = 2 * hw

        # (y, hh) are sampled first from self.combinations_x_hw and converted to (y, h) by h = 2 * hh
        y, hh = self.combinations_x_hw[np.random.randint(
            0, len(self.combinations_x_hw))]
        h = 2 * hh

        return np.array([5, x, y, w, h])