# coding: utf-8
"""
Multidimensional data classes and function
"""

import numpy as np
import copy
import matplotlib.pyplot as plt


class MultiData():
    def __init__(self, length=None, dimension=None):
        self.length = length
        self.dimension = dimension
        self.X = None

    def create_data(self, sigma, change_point, change_mean):
        """
        Get the multidimensional data

        [Parameters]
        sigma <int> : variance of the data
        chanage_point <multi_list> : change points position
        change_mean <mutli_list> : mean of the segment

        [Returns]
        numpy.ndarray
        """

        X = np.empty((0,self.length))
        for i in range(self.dimension):
            X_full = np.array([])
            if len(change_point[i])-1 != len(change_mean[i]):
                print('変化点と平均があってません')
                exit()
            for j, t in enumerate(change_point[i]):
                if j == 0:
                    continue
                X_before = np.random.normal(loc=change_mean[i][j-1], scale=sigma, size=t - change_point[i][j-1])
                X_full = np.append(X_full, X_before)
            X = np.append(X,[X_full], axis=0)
        self.X = X
        return X

    def normalized_data(self):
        """
        Get the normalized data(Set variance of each series to 1)


        [Returns]
        numpy.ndarray
        """
        k = np.shape(self.X)[0]
        clone_X = copy.deepcopy(self.X)
        for i in range(k):
            clone_X[i,:] = clone_X[i,:] / np.std(clone_X[i,:])
        return clone_X

    def output_X(self):
        """
        Output data(csv file)

        """
        np.savetxt('input_X.csv',self.X, delimiter=',')
