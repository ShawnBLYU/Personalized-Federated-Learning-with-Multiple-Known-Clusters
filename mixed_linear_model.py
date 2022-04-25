# Experimenting under mixed linear models

import numpy as np
from sim_cluster_data import get_cluster

class LinearClient():
    def __init__(self, data):
        self.X = data[:, :-1]
        self.y = data[:, -1]
        self.num_data = self.X.shape[0]
        self.dim = self.X.shape[1]
        self.theta = np.zeros(self.dim)

    def learn_local_weight(self):
        # regress y on X using only local data
        theta_hat_d, _, _, _ = np.linalg.lstsq(self.X, self.y)
        self.theta_hat_d = theta_hat_d
        return theta_hat_d

    def step(self, step_size):
        grad = self.X.T @ self.X @ self.theta - self.X.T @ self.y
        self.theta -= step_size * grad
        return np.linalg.norm(grad)

    def get_weight(self):
        return self.theta
    
    def set_weight(self, theta):
        self.theta = theta

class LinearNetwork():
    def __init__(self, list_of_clients, lambdas, gammas, num_clients):
        # lambdas: list of lambdas in the original problem. 
        # gammas: list of gammas in the original problem.
        # num_clients: number of clients in each cluster.
        self.list_of_clients = np.array(list_of_clients)
        self.num_clients = num_clients
        self.num_clusters = len(num_clients)
        self.lambdas = np.array(lambdas)
        self.gammas = np.array(gammas)
        self.dim = list_of_clients[0].dim
        self.alphas = self.get_alphas()

    def get_client_cluster(self, client_index):
        # gets the index of a client and returns the cluster the client belongs
        # to.
        return get_cluster(self.num_clients, client_index)

    def get_cluster_clients(self, cluster_index):
        # given an index of clusters, return the indices of clients belonging 
        # to the cluster as a list
        if cluster_index < 0 or cluster_index >= len(self.num_clients):
            raise NameError("Client index out of bound")
        # the number of clients in the first j clusters
        agg_sum = np.cumsum(self.num_clients)
        if cluster_index == 0:
            lower = 0
            higher = agg_sum[0]
        else:
            lower = agg_sum[cluster_index - 1]
            higher = agg_sum[cluster_index]
        return np.arange(lower, higher).astype(int)

    # How to work alpha in this?
    def get_alphas(self):
        res = np.zeros(len(self.lambdas))
        for j in range(self.num_clusters):
            # print(self.get_cluster_clients(j))
            client_gammas = self.gammas[self.get_cluster_clients(j)]
            res[j] = self.lambdas[j] / (self.lambdas[j] + sum(client_gammas))
        return res