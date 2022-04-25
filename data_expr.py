# Experiment Script for the DMEF dataset

import pandas as pd
import numpy as np
import logging
import argparse

from uszipcode import SearchEngine
from joblib import dump, load

from mixed_logit_model import *
from l2gd import *

parser = argparse.ArgumentParser(
    description='l2gd experiments on dmef data'
)
parser.add_argument(
    '--lamb',
    help='setting the lambda uniformly',
    default=1.
)
parser.add_argument(
    '--gamma',
    help='setting the gamma uniformly',
    default=1.
)
parser.add_argument(
    '--num_iter',
    help='setting number of iterations of l2gd',
    default=2000

)
parser.add_argument(
    '--lr',
    help='setting learning rate of l2gd',
    default=1e-3
)
parser.add_argument(
    '--p',
    help='setting communication probability of l2gd',
    default=1e-1
)
parser.add_argument(
    '--array_id',
    help='array id',
    type=int,
    default=-99
)

parsed = parser.parse_args()
lamb_list = 10 ** np.linspace(-1, 1, 8)
gamma_list = 10 ** np.linspace(-1, 1, 8)
lr_list = np.array([1e-4, 1e-3, 1e-2, 1e-1])

if parsed.array_id != -99:
    array_id = int(parsed.array_id)
    lamb = lamb_list[array_id % 8]
    array_id = array_id // 8
    gamma = gamma_list[array_id % 8]
    array_id = array_id // 8
    lr = lr_list[array_id % 4]
    num_iter = parsed.num_iter
    p = parsed.p
else:
    lamb = parsed.lamb
    gamma = parsed.gamma
    num_iter = parsed.num_iter
    lr = parsed.lr
    p = parsed.p

log_loc = "/home/blyu/fl_multi_cluster/log/data_expr_{}_{}_{}_{}_{}.log".format(
    str(round(lamb, 3)).replace('.', 'dot'),
    str(round(gamma, 3)).replace('.', 'dot'),
    str(num_iter).replace('.', 'dot'),
    str(lr).replace('.', 'dot'),
    str(p).replace('.', 'dot')
)

logging.basicConfig(
    filename=log_loc,
    filemode='w',
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO)
logging.info('Script starts')

DATA_LOC = "/home/blyu/fl_multi_cluster/data/proc_data.pkl"
TRAIN_RATIO = 0.8
all_data = pd.read_pickle(DATA_LOC)

# formating the covariates
all_data['zip'] = all_data['zip'].str.decode('utf-8')
all_data['recency'] = all_data['recency'].dt.days
all_data['success'] = 2 * (all_data['success'].astype(int)) - 1

# filter so that each zip has at least 10 entries
full_obs_data = all_data.groupby('zip').filter(lambda x: x['id'].count() >= 10)

full_obs_data['is_train'] = 0
def train_test_split(grp, ratio=TRAIN_RATIO):
    num_train = np.round(grp.shape[0] * ratio).astype(int)
    train_indices = np.random.choice(
        np.arange(grp.shape[0]), num_train, replace=False
    ).astype(int)
    train_indicator = np.zeros(grp.shape[0])
    train_indicator[train_indices] = 1
    return train_indicator

full_obs_data['is_train'] = full_obs_data.groupby('zip').is_train.transform(train_test_split)

train_data_full_cov = full_obs_data[full_obs_data['is_train'] > 0]
test_data_full_cov = full_obs_data[full_obs_data['is_train'] == 0]

index_to_zip = full_obs_data['zip'].unique()
zip_code_search = SearchEngine(simple_zipcode=True)

# since zipcode includes APO, AE
def zip_to_state(z):
    try:
        return zip_code_search.by_zipcode(z).state_abbr
    except:
        return "other"

index_to_state = np.array([
    zip_to_state(z) for z in index_to_zip
])

new_index_order = np.argsort(index_to_state)
index_to_zip = index_to_zip[new_index_order]
index_to_state = index_to_state[new_index_order]
zip_to_index = {}
for i in range(len(index_to_zip)):
    zipcode = index_to_zip[i]
    zip_to_index[zipcode] = i

logging.info('data processed')

# cluster the clients based on zipcode
num_clients = np.array([
    np.sum(index_to_state == s) for s in np.sort(np.unique(index_to_state))
])

covariates = ['freq', 'recency']
response = ['success']

test_data_grouped = test_data_full_cov.groupby('zip')[covariates + response]
train_data_grouped = train_data_full_cov.groupby('zip')[covariates + response]

def log_transform_data(z, grouped=train_data_grouped):
    # input: a zipcode z
    # output: data with log transformed covariates and additional intercept
    raw_data = grouped.get_group(z).to_numpy(float)
    raw_data[:, :2] = np.log(raw_data[:, :2])
    raw_data = np.hstack((
        raw_data[:, :2],
        np.ones((raw_data.shape[0], 1)),
        raw_data[:, 2][:, np.newaxis]
    ))
    return raw_data

list_of_clients = [
    LogitClient(
        log_transform_data(z)
    ) for z in index_to_zip
]

test_data = [
    log_transform_data(z, grouped=test_data_grouped) for z in index_to_zip
]

network = LogitNetwork(
    list_of_clients,
    [lamb] * len(num_clients),
    [gamma] * np.sum(num_clients),
    num_clients
)


test_data_full = test_data_full_cov[covariates + response].to_numpy(float)
train_data_full = train_data_full_cov[covariates + response].to_numpy(float)

train_data_full[:, :-1] = np.log(train_data_full[:, :-1])
test_data_full[:, :-1] = np.log(test_data_full[:, :-1])

train_data_full = np.hstack((
    train_data_full[:, :-1],
    np.ones((train_data_full.shape[0], 1)),
    train_data_full[:, -1][:, np.newaxis]
))
test_data_full = np.hstack((
    test_data_full[:, :-1],
    np.ones((test_data_full.shape[0], 1)),
    test_data_full[:, -1][:, np.newaxis]
))

logging.info('data loaded')
logging.info('start training global model')

full_model = LR(
    penalty='none', solver='lbfgs', fit_intercept=False
).fit(train_data_full[:, :-1], train_data_full[:, -1])
dump(full_model, 'saved_full_model.joblib')
full_mod = load('saved_full_model.joblib')
full_mod.score(test_data_full[:, :-1], test_data_full[:, -1])

logging.info('global model trained.')
logging.info('start training local model.')

for c in list_of_clients:
    c.train_local_model()

logging.info('local model trained')
logging.info('start training l2gd')

L2GD_Comb(
    network,
    lr,
    p,
    num_iter,
    tol=1e-6
)

logging.info('l2gd trained')
logging.info('start evaluating cross entropy')

def calc_cross_entropy(i, which_theta='l2gd'):
    # calculates the cross entropy loss for the i-th client
    test_x = test_data[i][:, :-1]
    test_y = test_data[i][:, -1]
    # recenter test_y to 0, 1
    test_y_01 = (test_y + 1) / 2
    c = list_of_clients[i]

    if which_theta == 'l2gd':
        pred_success_prob = c.predict_prob(test_x)
    elif which_theta == 'local':
        pred_success_prob = c.predict_prob(test_x, which_theta='local')
    elif which_theta == 'global':
        pred_success_prob = full_mod.predict_proba(test_x)[:, 1]

    log_loss_vec = - (
        test_y_01 * np.log(pred_success_prob) +
        (1 - test_y_01) * np.log(1 - pred_success_prob)
    )

    return np.mean(log_loss_vec)

l2gd_losses = [calc_cross_entropy(i) for i in range(len(list_of_clients))]
local_losses = [calc_cross_entropy(i, 'local') for i in range(len(list_of_clients))]
global_losses = [calc_cross_entropy(i, 'global') for i in range(len(list_of_clients))]

l2gd_losses = np.array(l2gd_losses)
local_losses = np.array(local_losses)
global_losses = np.array(global_losses)

all_losses = np.vstack((
    l2gd_losses,
    local_losses,
    global_losses
))

out_loc = "/home/blyu/fl_multi_cluster/out/data_expr_{}_{}_{}_{}_{}.npy".format(
    str(round(lamb, 3)).replace('.', 'dot'),
    str(round(gamma, 3)).replace('.', 'dot'),
    str(num_iter).replace('.', 'dot'),
    str(lr).replace('.', 'dot'),
    str(p).replace('.', 'dot')
)

with open(out_loc, 'wb') as f:
    np.save(f, all_losses)

logging.info('cross entropy evaluated and logged')
logging.info('success. exiting.')