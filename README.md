# Personalized Federated Learning with Multiple Known Clusters
Python implementation for the submission "Personalized Federated Learning with Multiple Known Clusters"
## General Helper Functions
- `L2GD.py` implements of L2GD in the multi-cluster setting.
- `mixed_linear_models.py` implements a hierarchical linear regression model that can be trained via L2GD.
- `mixed_linear_models.py` implements a hierarchical logistic regression model that can be trained via L2GD.
## Code for Simulation Studies
- `sim_cluster_data.py` implements data simulation for generating data in a hierarchical linear regression model.
- `sim_data.py` performs the experiments on simulated data.
## Code for DMEF Customer Lifetime Value Modeling Task
- `data_proc.py` implements data processing for the raw data obtained directly from DMEF.
- `data_expr.py` performs the experiments on the real-world task, i.e. DMEF Customer Lifetime Value Modeling Task.
