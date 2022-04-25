# Script for simulating the hierarchical Bayes Linear regresion model.

import numpy as np
VERBOSE = False

# Simulating a hierarchical Bayes Linear regression model
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

def get_data(client_num_datapoints,
             client_index):
    # given an array recording the number of data points at each client and
    # the client index, returns the starting and ending indices of the clients'
    # data in the data chunk as a tuple
    if client_index < 0 or client_index >= len(client_num_datapoints):
        raise NameError("Client index out of bound")
    agg_sum = np.cumsum(client_num_datapoints)
    if client_index == 0:
        return (0, agg_sum[0])
    else:
        return (agg_sum[client_index - 1], agg_sum[client_index])

def sim_data(theta_bar,
             num_clusters,
             num_clients,
             dimensions,
             cluster_weight_var,
             client_weight_var,
             client_y_var,
             client_num_datapoints,
             sim_data_location):
    # theta_bar: center weight for all cluster centers, \Bar{\theta}^*
    #     in the paper
    # num_clusters: number of clusters
    # num_clients: an array giving the number of clients for each cluster.
    #     Require length(num_clients) == num_clusters
    # dimensions: the weight vector's dimension
    # cluster_weight_var: variance of the weights for cluster centers
    # client_weight_var: an array recording the variance of client weights
    #     in each cluster. Require length(client_weight_var) == num_clusters.
    # client_y_var: variance of y at each client. Require sum(num_clients) ==
    #     length(client_y_var)
    # client_num_datapoints: number of data points at each client. Require
    #     sum(num_clients) == length(client_num_datapoints)

    # Saves the generated data as a pickle file
    # X: random matrix with iid N(0, 1) entries
    # y: response vector
    # theta_bar: theta_bar
    # cluster_center_weights: theta_bar_j
    # client_weights: theta_i
    # num_clients: number of clients in each cluster. sum(num_clients) == n
    # client_num_datapoints: number of observations for each client

    # generating the cluster center weights
    cluster_center_weights = theta_bar + (np.sqrt(cluster_weight_var) *
        np.random.randn(num_clusters, dimensions))
    if VERBOSE:
        print(cluster_center_weights)
    # generating the clients weights
    client_weights = np.zeros((np.sum(num_clients), dimensions))
    for i in range(np.sum(num_clients)):
        cluster_index = get_cluster(num_clients, i)
        client_weights[i] = (cluster_center_weights[cluster_index] +
            np.sqrt(client_weight_var[cluster_index]) *
            np.random.randn(dimensions))
    if VERBOSE:
        print(client_weights)
        print(client_weights.shape)
    # generating the samples at each client
    X = np.zeros((np.sum(client_num_datapoints), dimensions))
    y = np.zeros((np.sum(client_num_datapoints)))
    for i in range(np.sum(num_clients)):
        data_start, data_end = get_data(client_num_datapoints, i)
        X[data_start : data_end, :] = np.random.randn(
            client_num_datapoints[i], dimensions
        )
        y[data_start : data_end] = (
            X[data_start : data_end, :] @ client_weights[i]
        ) + np.random.randn(data_end - data_start) * np.sqrt(client_y_var[i])
        if VERBOSE:
            print('Simulated data for client {}'.format(i))
            print(X[data_start : data_end, :])
            print(y[data_start : data_end])
    if VERBOSE:
        print(X)
        print(y)
    with open(sim_data_location, 'wb') as f:
        np.savez(
            f,
            theta_bar=theta_bar,
            cluster_center_weights=cluster_center_weights,
            client_weights=client_weights,
            num_clients=num_clients,
            client_num_datapoints=client_num_datapoints,
            X=X,
            y=y
        )
        f.close()
    return {
        'theta_bar':theta_bar,
        'cluster_center_weights':cluster_center_weights,
        'client_weights':client_weights,
        'num_clients':num_clients,
        'client_num_datapoints':client_num_datapoints,
        'X':X,
        'y':y
    }
