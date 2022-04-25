import numpy as np

def L2GD_Comb(network, step_size, p, max_iter, tol=1e-6):
    for t in range(max_iter):
        xi = np.random.binomial(1, p)
        
        if xi == 0:
            
            avg_grad_norm = 0
            for d in network.list_of_clients:
                avg_grad_norm += d.step(step_size / (1 - p))
            avg_grad_norm = avg_grad_norm / (len(network.list_of_clients))

            if avg_grad_norm <= tol:
                return t

        else:
            
            local_avgs = np.zeros((network.num_clusters, network.dim))
            sum_params = np.zeros(network.dim) # weighted sum of learned params
            sum_weights = 0 # weights of the average

            for j in range(network.num_clusters):
                client_indices = network.get_cluster_clients(j)
                # print(client_indices)
                client_gammas = network.gammas[client_indices]
                clients = network.list_of_clients[client_indices]
                client_weights = np.array([c.theta for c in clients])
                
                local_avgs[j] = (
                    client_gammas[np.newaxis, :]
                    @ client_weights
                    / np.sum(client_gammas)
                )

                sum_params += (
                    network.alphas[j]
                    * (client_gammas[np.newaxis, :]
                    @ client_weights)[0, :]
                )

                sum_weights += network.alphas[j] * np.sum(client_gammas)
            
            # Implement vectorized version of global aggregation
            global_avg = sum_params / sum_weights

            # broadcast aggregated weight back to the clients
            for j in range(network.num_clusters):
                client_indices = network.get_cluster_clients(j)
                alpha_j = network.alphas[j]
                w_j = alpha_j * global_avg + (1 - alpha_j) * (local_avgs[j])
                for i in client_indices:
                    c_i = network.list_of_clients[i]
                    gamma_i = network.gammas[i]
                    c_i.theta = (
                        (1 - step_size * gamma_i / p) * c_i.theta
                        + step_size * gamma_i / p * w_j
                    )
    return t


