from typing import NamedTuple, List

import glob
import numpy as np
import itertools
import csv

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


def parse_files_in_data(data_name: str, limit: int = None) -> np.ndarray:
    file_list = glob.glob(f'./data/{data_name}/*.npz')
    traj_collections = list(map(parse_one_file, itertools.islice(file_list, limit)))
    #print(traj_collections)
    tc =  concat_traj_collections(traj_collections)
    return tc.feat[:, :, :1024].reshape(-1, 50, 32, 32)

def extract_csv_episodes(reader: csv.DictReader):
    feat_sequence = []
    i = iter(reader)
    row = next(i)
    while True:
        try:
            # feat_sequence.append(row['Start_state'])
            current_ep_num = row['Episode']
            while row["Episode"] == current_ep_num:
                feat_sequence.append(row['End_state'])
                row = next(i)
            yield feat_sequence
            feat_sequence = []
        except StopIteration:
            yield feat_sequence
            break


def process_one_feature(feat: str):
    return (np.frombuffer(bytes(feat[2:-1], 'utf8'), np.uint8, 1024) - 48).reshape(32, 32)

def parse_csv(name: str) -> List[np.ndarray]:
    with open(f'./csv_data/{name}.csv', 'r') as file:
        reader = csv.DictReader(file)
        episodes = list(extract_csv_episodes(reader))
        return [np.asarray(list(map(process_one_feature, ep))) for ep in episodes]
