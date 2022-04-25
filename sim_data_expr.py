import mixed_linear_model
from sim_cluster_data import sim_data, get_data, get_cluster
import l2gd

import numpy as np
import sys
np.random.seed(0)
num_samples = [1, 5, 10, 25, 50, 100, 200]
which_n = 1
n = num_samples[which_n]
d = 20 # dimensions
clients_per_cluster = 20 * [20] # number of clients per cluster
num_clusters = len(clients_per_cluster)
num_clients = sum(clients_per_cluster)

theta_bar = np.zeros(d)

tmp = sim_data(
    theta_bar,
    num_clusters,
    clients_per_cluster,
    d,
    1, 
    np.ones(num_clusters),
    np.ones(num_clients),
    [n] * num_clients,
    "tmp.npz"
)

sim_X = tmp['X']
sim_y = tmp['y']
sim_Xy = np.hstack((sim_X, sim_y[:, np.newaxis]))
sim_num_datapoints = tmp['client_num_datapoints']
sim_theta_bar = tmp['theta_bar']
sim_cluster_center_weights = tmp['cluster_center_weights']
sim_client_weights = tmp['client_weights']
sim_num_clients = tmp['num_clients']

clients = [
    mixed_linear_model.LinearClient(
        sim_Xy[get_data(sim_num_datapoints, i)[0] : get_data(sim_num_datapoints, i)[1], :]
    ) for i in range(num_clients)
]

network = mixed_linear_model.LinearNetwork(
    clients,
    [1.] * num_clusters,
    [1.] * num_clients,
    sim_num_clients   
)

l2gd.L2GD_Comb(
    network,
    1e-4,
    0.1,
    50000
)

gls_errs = np.array([
    np.linalg.norm(network.list_of_clients[i].theta - sim_client_weights[i]) for
        i in range(len(network.list_of_clients))
])

for c in network.list_of_clients:
    c.learn_local_weight()

local_errs = np.array([
    np.linalg.norm(network.list_of_clients[i].theta_hat_d - sim_client_weights[i]) for
        i in range(len(network.list_of_clients))
])

glob_weight, _, _, _ = np.linalg.lstsq(sim_X, sim_y)

global_errs = np.array([
    np.linalg.norm(glob_weight - sim_client_weights[i]) for
        i in range(len(network.list_of_clients))
])

candidate_lambs = np.linspace(0.01, 2, 20)
sc_errs_all_lambs = np.zeros((20, num_clients))
for i in range(len(candidate_lambs)):
    lamb = candidate_lambs[i]
    helper_matrix = np.array([
        np.linalg.inv(c.X.T @ c.X + lamb * np.eye(d))
            for c in network.list_of_clients
    ])

    sc_center = np.linalg.inv(
        np.eye(d) - lamb * np.mean(helper_matrix, axis=0)
    ) @ np.mean([
        helper_matrix[i] @ (
            network.list_of_clients[i].X.T @ network.list_of_clients[i].X
        ) @ network.list_of_clients[i].theta_hat_d for i in range(
            len(network.list_of_clients))
    ], axis=0)
    print(sc_center)
    sc_weights = np.array([
        helper_matrix[i] @ (
            network.list_of_clients[i].X.T @ network.list_of_clients[i].X
        ) @ network.list_of_clients[i].theta_hat_d + lamb * sc_center
        for i in range(len(network.list_of_clients))
    ])
    print(sc_weights)
    sc_errs_all_lambs[i, :] = np.sqrt(((sc_weights - sim_client_weights) ** 2).sum(axis=1))
    print(np.mean(sc_errs_all_lambs, axis=1))

best_lamb_idx = np.argmin(np.mean(sc_errs_all_lambs, axis=1))
sc_errs = sc_errs_all_lambs[best_lamb_idx, :]
print(sc_errs)

res = np.vstack((
    gls_errs,
    local_errs,
    global_errs,
    sc_errs
))

out_loc = "/home/blyu/fl_multi_cluster/sim_out/new_sim_expr_{}.npy".format(
    str(n)
)

with open(out_loc, 'wb') as f:
    np.save(f, res)

