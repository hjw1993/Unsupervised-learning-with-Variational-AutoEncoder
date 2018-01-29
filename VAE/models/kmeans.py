"""Implements the k-means algorithm.
"""

import numpy as np
import scipy
from scipy import stats


class KMeans(object):
    def __init__(self, n_dims, n_components=10, max_iter=100):
        """Initialize a KMeans GMM model
        Args:
            n_dims(int): The dimension of the feature.
            n_components(int): The number of cluster in the model.
            max_iter(int): The number of iteration to run EM.
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter

        # Randomly initialize model parameters using Gaussian distribution of 0
        # mean and unit variance.
        cen=[]
        for i in range (0,n_components):
            mu, sigma = 0, 1
            rarray = np.random.random(size=n_dims)
            cen.append(rarray)

        self._mu = np.array(cen)  # np.array of size (n_components, n_dims)




    def fit(self, x):
        """Runs EM step for max_iter number of times.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
            ndims).
        """
        for i in range(self._max_iter):
            r_ik=self._e_step(x)

            self._m_step(x,r_ik)

        pass

    def _e_step(self, x):
        """Update cluster assignment.

        Computes the posterior probability of p(z|x), z is the latent cluster
        variable.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            r_ik(numpy.ndarray): Array containing the cluster assignment of
                each example, dimension (N,).
        """
        l=[]
        for i in range(len(x)):
            dis=[]
            for j in range(len(self._mu)):
                dis.append(np.sqrt(np.sum(np.square(x[i] - self._mu[j])))  )

            l.append(dis.index(min(dis)))


        r_ik = None
        r_ik=np.array(l)
        return r_ik

    def _m_step(self, x, r_ik):
        """Update cluster mean.

        Updates self_mu according to the cluster assignment.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            r_ik(numpy.ndarray): Array containing the cluster assignment of
                each example, dimension (N,).
        """
        for i in range(self._n_components):
            self._mu[i] = np.mean(x[r_ik == i], axis=0)

        pass

    def get_posterior(self, x):
        """Computes cluster assignment.

        Computes the posterior probability of p(z|x), z is the latent cluster
        variable.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            r_ik(numpy.ndarray): Array containing the cluster assignment of
            each example, dimension (N,).
        """

        l = []
        for i in range(len(x)):
            dis = []
            for j in range(len(self._mu)):
                dis.append(np.sqrt(np.sum(np.square(x[i] - self._mu[j]))))
            l.append(dis.index(min(dis)))



        r_ik = None
        r_ik = np.array(l)
        return r_ik


    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.

        For each cluster, find the most common digit using the provided (x,y)
        and store it in self.cluster_label_map.

        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.

        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """
        l=[]
        c_l=[[0 for i in range(10)]for j in range(10)]
        for i in range(len(x)):
            dis = []
            for j in range(len(self._mu)):
                dis.append(np.sqrt(np.sum(np.square(x[i] - self._mu[j]))))
            l.append(dis.index(min(dis)))
            c_l[dis.index(min(dis))][int(y[i][0])]+=1


        self.cluster_label_map = []
        for i in range(len(c_l)):
            self.cluster_label_map.append(c_l[i].index(max(c_l[i])))
        print(self.cluster_label_map)
        pass

    def supervised_predict(self, x):
        """Predict a label for each example in x.
        Find the get the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
                x, dimension (N,)
        """

        r_ik = self.get_posterior(x)
        y_hat = []

        for i in range(len(x)):

            y_hat.append(self.cluster_label_map[r_ik[i]])


        y_hat=np.array(y_hat)


        return np.array(y_hat)
