# Experimenting under mixed linear models

import numpy as np
from sklearn.linear_model import LogisticRegression as LR

def get_cluster(num_clients,
                client_index):
    # given an array recording the number of clients at each cluster and an
    # index of the client, returns the index of the cluster that the client
    # belongs to.
    if client_index < 0 or client_index >= np.sum(num_clients):
        raise NameError("Client index out of bound")
    # the number of clients in the first j clusters
    agg_sum = np.cumsum(num_clients)
    return np.argmax(client_index < agg_sum)


class LogitClient():
    def __init__(self, data):
        self.X = data[:, :-1]
        self.y = data[:, -1]
        self.num_data = self.X.shape[0]
        self.dim = self.X.shape[1]
        self.theta = np.zeros(self.dim)
        self.theta_hat_d = np.zeros(self.dim)

    def train_local_model(self):
        if len(np.unique(self.y)) == 1:
            # make the intercept fit the label
            self.theta_hat_d[-1] = self.y[0]
        else:
            lr = LR(fit_intercept=False, penalty='none')
            lr.fit(self.X, self.y)
            self.theta_hat_d = lr.coef_[0]
        return self.theta_hat_d
        
    def step(self, step_size):
        logit = np.exp(-self.y * (self.X @ self.theta))
        grad = (((logit)/(1 + logit) * (-self.y)) * self.X.T).T.mean(axis=0)
        self.theta -= step_size * grad
        return np.linalg.norm(grad)

    def predict_prob(self, X, which_theta='l2gd'):
        # returns the probability of X being class 1
        if which_theta == 'l2gd':
            theta = self.theta
        elif which_theta == 'local':
            theta = self.theta_hat_d
        else:
            raise ValueError('wrong arg for theta')
        return 1 / (1 + np.exp(- X @ theta))

    def get_weight(self):
        return self.theta
    
    def set_weight(self, theta):
        self.theta = theta

class LogitNetwork():
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