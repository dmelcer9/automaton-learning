from typing import NamedTuple, List

import glob
import numpy as np
import itertools


class TrajCollection(NamedTuple):
    feat: np.ndarray
    reward: np.ndarray
    action: np.ndarray
    stochastic_feat: np.ndarray
    deterministic_feat: np.ndarray


def parse_one_file(path):
    file = np.load(path)
    #print(file['reward'].shape)
    return TrajCollection(file['feat'], file['reward'], file['action'], file['stochastic_feat'], file['deterministic_feat'])


def concat_traj_collections(traj_collections: List[TrajCollection]):
    feat = np.concatenate([traj_collection.feat for traj_collection in traj_collections])
    reward = np.concatenate([traj_collection.reward for traj_collection in traj_collections])
    action = np.concatenate([traj_collection.action for traj_collection in traj_collections])
    stochastic_feat = np.concatenate([traj_collection.stochastic_feat for traj_collection in traj_collections])
    deterministic_feat = np.concatenate([traj_collection.deterministic_feat for traj_collection in traj_collections])
    return TrajCollection(feat, reward, action, stochastic_feat, deterministic_feat)


def parse_files_in_data(data_name: str, limit: int = None) -> TrajCollection:
    file_list = glob.glob(f'./data/{data_name}/*.npz')
    traj_collections = list(map(parse_one_file, itertools.islice(file_list, limit)))
    #print(traj_collections)
    return concat_traj_collections(traj_collections)

