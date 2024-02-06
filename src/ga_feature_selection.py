"""
This module is used to test how feature selection for training Viola Jones cascade classifier can be done using genetic algorithm.
Feature selection is done before training each stage of the cascade classifier and the selected features are used to train the strong classifier for that stage.
"""

import os
import time

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import pandas as pd
import pygad
import xgboost as xgb
import lightgbm as lgb
# import matplotlib.pyplot as plt
import sklearn

import utils
import ga_population_generator
import features_viola_haar as ft
import ga_fitness_result_store as ga_fitness_store
import ga_mutation

# basic steps:
# 0.a load the training and validation data for the current stage
# 0.b initialize ga_instance with the relevant parameters
# 1. generate initial population
# 2. generate feature vectors based on each individual (chromosome) of the population
# 3. calculate fitness of each individual by training an XGBClassifier with the selected features (training done using histogram based methods for faster training)
# 4. Evaluate the fitness of each individual by looking at the accuracy and feature importance
# 5. Store the evaluated fitness of each individual across generations to compute the average fitness of each individual
# 6. Select the best individuals from the population based on the fitness for crossover
# 7. Perform crossover and mutation to generate new population


def load_data(data_df):
    # filter to keep only the train folds
    train_df = data_df[data_df['fold'].isin({3, 4, 5, 6})]
    val_df = data_df[data_df['fold'].isin({7, 8, 9, 10})]

    X_train = np.array(
        [utils.load_image(img_path) for img_path in train_df['img_path']])
    y_train = np.array(train_df['label'])

    X_val = np.array(
        [utils.load_image(img_path) for img_path in val_df['img_path']])
    y_val = np.array(val_df['label'])

    return X_train, y_train, X_val, y_val


def get_split(data):
    """
    This function splits X, y into train and val sets
    """
    img_int, nf, y = data

    # get indices of train and val sets
    n = len(y)
    train_ratio = 0.7
    num_train = int(n * train_ratio)
    num_val = n - num_train
    indices = np.array([1] * num_train + [0] * num_val)

    np.random.shuffle(indices)

    # split the data into train and val sets
    xis_train = img_int[indices == 1]
    xis_val = img_int[indices == 0]
    nf_train = nf[indices == 1]
    nf_val = nf[indices == 0]
    y_train = y[indices == 1]
    y_val = y[indices == 0]

    return xis_train, nf_train, y_train, xis_val, nf_val, y_val


def custom_on_start(ga_instance):
    # customize the generation of the initial population since our chromosome space (space of all haar features) is not supported by pygad
    # generate a random population of size sol_per_pop

    # initialize the HaarFeatureSampler object to generate initial population
    haar_feature_sampler = ga_instance.haar_feature_sampler

    # generate initial population
    initial_population = haar_feature_sampler.sample(ga_instance.sol_per_pop)

    # set the initial population
    ga_instance.population = initial_population
    ga_instance._custom_scores_list = []


def custom_mutation(offspring_crossover, ga_instance):
    """
    custom mutation function to mutate the offspring got after crossover
    """

    haar_feature_mutator = ga_instance.haar_feature_mutator

    offspring_crossover = haar_feature_mutator.mutate(offspring_crossover)

    return offspring_crossover


def custom_on_fitness(ga_instance, population_fitness):
    # calculate the average fitness of top 5% of all the individuals evaluated till now

    K = ga_instance.train_args.maxWeakCount // 2
    avg_score = ga_instance.fitness_store.get_avg_score(top_k=K)
    ga_instance._custom_scores_list.append(avg_score)

    # check if the last_generation_fitness sums to 0 as RWS will give DivideByZero error
    if np.sum(population_fitness) == 0:
        a = 1
        # ga_instance.last_generation_fitness = np.ones(len(population_fitness))


def fitness_func_batch_lgb(ga_instance, solutions, solutions_indices):
    """
    Batch fitness function allows us to calculate the fitness of all the individuals in the population in a single function call.
    This is required since we will be using the performance of the trained XGBClassifier on the features (individuals) to calculate the fitness of the individuals.
    """
    # get a random split of the data into train and val sets
    xis_train, nf_train, y_train, xis_val, nf_val, y_val = get_split(
        ga_instance.data)

    # compute the features (individuals) on the train and val sets to generate train and val feature vectors
    computed_features_train = ft.compute_features_all_imgs(
        solutions, xis_train, nf_train)
    computed_features_val = ft.compute_features_all_imgs(
        solutions, xis_val, nf_val)
    # transpose the computed features to have shape (n_samples, n_features)
    computed_features_train = computed_features_train.T
    computed_features_val = computed_features_val.T

    # fit the XGBClassifier on the train set and evaluate the performance on the val set
    # early stopping is used to prevent overfitting

    stage = ga_instance.current_stage
    n_estimators = ga_instance.train_args.LGBM_NUM_ESTIMATORS
    early_stopping_rounds = max(5, int(n_estimators * 0.1))

    # define the LightGBM classifier
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=1,
        max_leaves=2,
        max_bin=63,
        objective='binary',
        learning_rate=1,
        force_row_wise=True,
        metric='auc',
        early_stopping_round=early_stopping_rounds,
    )

    start_time = time.time()
    # fit the XGBClassifier on the train set and evaluate the performance on the val set
    lgb_clf.fit(computed_features_train,
                y_train,
                eval_set=[(computed_features_val, y_val)], verbose=False)
    end_time = time.time()
    print(f"Time taken to train LGBMClassifier: {end_time - start_time}")

    # calculate the fitness of each individual by looking at the accuracy and feature importance
    val_auc = lgb_clf.evals_result_['valid_0']['auc'][-1]
    feature_importances = lgb_clf.booster_.feature_importance(
        importance_type='gain')
    
    # create feature importance dictionary with key as "f{idx}" and value as feature importance
    feature_importances_dict = {
        f"f{idx}": importance
        for idx, importance in enumerate(feature_importances)
    }

    # update the fitness of inidividuals in the FitnessStore object of the GA instance
    ga_instance.fitness_store.update_fitness_scores(solutions, val_auc,
                                                    feature_importances_dict)

    # return the fitness of the individuals from the FitnessStore object
    return ga_instance.fitness_store.get_fitness_scores(solutions)


def select_features(img_int, nf, y, args, stage, ga_config):
    """
    Select features for training the strong classifier for the given stage of the cascade classifier.
    """

    ga_instance = pygad.GA(
        num_genes=ga_config.SIZE,
        # num_generations=50,
        num_generations=ga_config.NUM_GENERATIONS,
        # num_parents_mating=100,
        num_parents_mating=ga_config.NUM_PARENTS_MATING,
        # sol_per_pop=FEATURES_PER_GENERATION,
        sol_per_pop=ga_config.FEATURES_PER_GENERATION,
        # keep_parents=-1,
        # keep_elitism=30,
        # keep_elitism=1,
        keep_elitism=ga_config.KEEP_ELITISM,
        # parent_selection_type="rws",
        parent_selection_type=ga_config.PARENT_SELECTION_TYPE,
        # crossover_type="scattered",
        crossover_type=ga_config.CROSSOVER_TYPE,
        mutation_type=custom_mutation,
        # mutation_probability=0.2,
        mutation_probability=ga_config.MUTATION_PROBABILITY,
        # fitness_func=fitness_function_factory(weights, profits, capacity),
        # fitness_func=lambda x, y, z: x,
        # fitness_batch_size=FEATURES_PER_GENERATION,
        fitness_batch_size=ga_config.FEATURES_PER_GENERATION,
        # fitness_func=fitness_func_batch,
        fitness_func=fitness_func_batch_lgb,
        on_start=custom_on_start,
        on_fitness=custom_on_fitness,
        # init_range_low=0,
        # init_range_high=2,
        # random_mutation_min_val=0,
        # random_mutation_max_val=1,
        #    mutation_by_replacement=True,
        gene_type=int)

    # set the data, config and current stage to the ga instance so that it can be used by the fitness function to train the XGBClassifier
    ga_instance.data = (img_int, nf, y)
    ga_instance.train_args = args
    ga_instance.current_stage = stage

    # initialize the HaarFeatureSampler object to generate initial population and to get mutated individuals
    ga_instance.haar_feature_sampler = ga_population_generator.HaarFeatureSampler(
    )

    # initialize the HaarFeatureMutator object to mutate the individuals
    ga_instance.haar_feature_mutator = ga_mutation.HaarFeatureMutator(
        ga_instance)

    # initialize the FitnessStore object to store the fitness values of the individuals
    ga_instance.fitness_store = ga_fitness_store.FitnessStore(ga_instance)

    # run the GA
    ga_instance.run()

    # plt.figure()
    # plt.plot(ga_instance._custom_scores_list)
    # plt.savefig(f"temp/avg_fitness_top_k_features_stage_{stage}.jpg")

    # return the top K features
    top_k_features = ga_instance.fitness_store.get_best_features(K=1000)

    return top_k_features, ga_instance
