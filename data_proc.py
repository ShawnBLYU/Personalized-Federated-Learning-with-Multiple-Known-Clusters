# Data processing script for the DMEF dataset.

DATA_LOC = "/home/research/fl_multi_cluster/data/"

import pandas as pd
import numpy as np
import logging

# setting up logging scripts
logging.basicConfig(
    filename='new_proc_data.log',
    filemode='w',
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO)

logging.info('Script starts')

appeal_loc = DATA_LOC + "appeal.sas7bdat"
donor_loc = DATA_LOC + "donor.sas7bdat"
trans_loc = DATA_LOC + "trans.sas7bdat"

appeal = pd.read_sas(appeal_loc)
donor = pd.read_sas(donor_loc)
trans = pd.read_sas(trans_loc)

# memory saving
trans = trans.drop(['source', 'center'], axis=1)
appeal = appeal.drop('source', axis=1)

logging.info('dataset loaded')

appeal['id'] = appeal['id'].astype(int)
donor['id'] = donor['id'].astype(int)
trans['id'] = trans['id'].astype(int)

# Consider only donors with appeal, donor, and transaction histories
complete_info_ids = pd.Series(list(
    set(appeal['id']) & set(trans['id']) & set(donor['id'])
))
# filter and trans so that they only contain entries with complete info
appeal = appeal.loc[appeal['id'].isin(complete_info_ids)]
trans = trans.loc[trans['id'].isin(complete_info_ids)]
donor = donor.loc[donor['id'].isin(complete_info_ids)]

logging.info('dataset filtered')

tmp = appeal.merge(donor, on='id', how='inner')
tmp = tmp.loc[tmp['appdate'] > tmp['firstgift']]
tmp.index = pd.RangeIndex(len(tmp.index))

# clean out the donor dataframe from memory
donor = donor.drop(donor.index, inplace=True)

tmp['success'] = np.nan
tmp['recency'] = np.nan
tmp['freq'] = np.nan
tmp['zip'] = np.nan

# testing on a smaller batch
# tmp = tmp.iloc[:100, :]

tmp['next_appdate'] = tmp.groupby('id')['appdate'].transform(
    lambda x: x.shift(-1)
)

logging.info('added next_appdate to list')

def add_response(r):
    giftdates = trans.loc[trans['id'] == r['id'], 'giftdate']
    if pd.isnull(r['next_appdate']):
        num_donations = (giftdates >= r['appdate']).sum()
    else:
        num_donations = (
            (giftdates >= r['appdate']) & (giftdates < r['next_appdate'])
        ).sum()
    r['success'] = (num_donations > 0)
    trans_for_id = trans.loc[trans['id'] == r['id']]
    curr_appdate = r['appdate']
    last_trans_date = trans_for_id.loc[
        trans_for_id['giftdate'] < curr_appdate, 'giftdate'
    ].iloc[-1]
    r['zip'] = trans_for_id.loc[
        trans_for_id['giftdate'] < curr_appdate, 'zip'
    ].iloc[-1]
    r['freq'] = np.sum(
        trans_for_id['giftdate'] < curr_appdate
    )
    r['recency'] = curr_appdate - last_trans_date

    return r

logging.info('starts adding responses')
tmp_additional_covs = tmp.apply(add_response, axis=1)
logging.info('responses are added')
tmp_additional_covs.to_pickle(
    'new_proc_data.pkl'
)
logging.info('responses are saved')
logging.info('success. exiting now')