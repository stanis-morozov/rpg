#!/usr/bin/env python3

import numpy as np
import os
from catboost import Pool, CatBoostRegressor
from os.path import join

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

class RelevanceModel:
    def __init__(self, dataset, mode, datapath=None, modelpath=None):
        """
        A wrapper for Collections and Video Catboost models
        :param dataset: dataset name, possible values ['collections', 'video']
        :param mode: possible values ['train', 'test']
        :param datapath: path to the data folder
        :param modelpath: path to Catboost model corresponding to the dataset
        """
        assert dataset in ['collections', 'video'], "dataset should be in ['collections', 'video']"
        assert mode in ['train', 'test'], "mode should be in ['train', 'test']"

        if datapath is None:
            datapath = join(dataset, 'data')
        if modelpath is None:
            modelpath = join(dataset, 'model.bin')
        suffix = '_' + mode + '.fvecs'
        files = []
        for r, d, f in os.walk(join(datapath, 'pairwise')):
            for filename in f:
                if suffix in filename:
                    files.append(filename)

        self.pairwise_features = [f.replace(suffix, '') for f in files]
        self.user_features = open(join(datapath, 'user_features_list.txt')).read().split()
        self.user_feature_values = mmap_fvecs(join(datapath, 'user_features' + suffix))

        self.item_features = open(join(datapath, 'item_features_list.txt')).read().split()
        self.item_feature_values = mmap_fvecs(join(datapath, 'item_features.fvecs'))

        self.pairwise_feature_values = {}
        for ftr in self.pairwise_features:
            data = mmap_fvecs(os.path.join(datapath, 'pairwise', ftr + suffix))
            self.pairwise_feature_values[ftr] = data

        self.features_list = self.pairwise_features + self.item_features + self.user_features

        self.model = CatBoostRegressor()
        self.model.load_model(modelpath)

        self.additional_names = list(set(self.model.feature_names_) - set(self.features_list))
        self.additional_values = np.zeros((len(self.additional_names),))

        if dataset == 'collections':
            assert len(self.additional_names) == 0, "Looks like the data was corrupted. Verify that the dataset has been fully loaded"
        elif dataset == 'video':
            assert len(self.additional_names) == 497, "Looks like the data was corrupted. Verify that the dataset has been fully loaded"

        self.features_list += self.additional_names

    def get_scores(self, indices):
        """
        Evaluates model scores for given (user, item) ids
        :param indices: list of (user, item) ids pairs
        :return: Numpy array of score values
        """
        values = []
        np_indices = np.array(indices)
        for ftr in self.pairwise_features:
            values.append(self.pairwise_feature_values[ftr][np_indices[:, 0], np_indices[:, 1]])
        values = np.stack(values)
        
        i_values = []
        for i, ftr in enumerate(self.item_features):
            i_values.append(self.item_feature_values[np_indices[:, 1], i])
        i_values = np.stack(i_values)

        u_values = []
        for j, ftr in enumerate(self.user_features):
            u_values.append(self.user_feature_values[np_indices[:, 0], j])
        u_values = np.stack(u_values)

        add_values = np.repeat(self.additional_values, np_indices.shape[0]).reshape(-1, np_indices.shape[0])

        values = np.vstack((values, i_values, u_values, add_values))

        pool = Pool(values.T, feature_names=self.features_list)
        return self.model.predict(pool)

if __name__ == '__main__':
    # demo: use pre-trained catboost model to generate predictions for (user, item) pairs 
    # make sure they are close enough to the values in the precomputed dataset
    for dataset in ['collections', 'video']:
        for mode in ['train', 'test']:
            print('dataset: ', dataset, ', mode: ', mode, sep='')
            model = RelevanceModel(dataset, mode)

            n = 100
            ids = []
            for i in range(n):
                ids.append((np.random.randint(10**3), np.random.randint(10**6)))

            res = model.get_scores(ids)

            precomputed_scores = np.fromfile(join(dataset, 'data/model_scores/scores_' + mode + '.bin'), dtype=np.float32).reshape((10**6, 10**3)).T
            ans = precomputed_scores[np.array(ids)[:, 0], np.array(ids)[:, 1]]

            print('Difference with precomputed data:', np.linalg.norm(res - ans) / np.linalg.norm(ans))
