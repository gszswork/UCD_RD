import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import warnings

#warnings.filterwarnings("ignore", category=UserWarning)

class UCDDataset(nn.Module):
    def __init__(self, fold_x, data_path, seq_len):
        # seq_len: S in the (S, N, E) shape of the data
        self.fold_x = fold_x
        self.data_path = data_path
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.fold_x) 
    
    def find_all_paths(self, node_features, edge_index, root=0):
        n = node_features.size(0)
        adj_list = [[] for _ in range(n)]
        
        # Convert edge_index to an adjacency list
        for src, dst in edge_index.t().tolist():
            adj_list[src].append(dst)

        def dfs(node, path, paths):
            path.append(node)
            if len(adj_list[node]) == 0:  # If no children, it's a leaf node
                paths.append(path.copy())
            else:
                for neighbor in adj_list[node]:
                    dfs(neighbor, path, paths)
            path.pop()
        
        all_paths = []
        dfs(root, [], all_paths)

        def path_max_pool(node_features, paths):
            max_pool = []
            for path in paths:
                path_features = node_features[path]
                max_pool.append(path_features.max(dim=0).values)
            return max_pool
        
        all_paths = path_max_pool(node_features, all_paths)

        return all_paths


    def __getitem__(self, index):
        id = self.fold_x[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)

        x_features=torch.tensor([item.detach().numpy() for item in list(data['x'])],dtype=torch.float32)
        # x_features.shape = [seqlen, 768]
        seqlen = x_features.shape[0]
        rootindex = int(data['rootindex'])
        edgeindex = torch.LongTensor(data['edgeindex'])
        all_paths = self.find_all_paths(x_features, edgeindex, root=rootindex)
        # print(len(all_paths), all_paths)

        if seqlen < self.seq_len:
            # If seqlen < S, pad the tensor to shape [S, 768]
            pad_size = self.seq_len - seqlen
            x_features = nn.functional.pad(x_features, (0, 0, 0, pad_size))
        else:
            # If seqlen > S, crop the tensor to shape [S, 768]
            x_features = x_features[:self.seq_len, :]

        return Data(x=x_features.unsqueeze(dim=0), y=torch.LongTensor([int(data['y'])]))

if __name__ == '__main__':
    data_path = '../data/in-domain/Twittergraph'
    twitter_in_ids = np.load('../data/twitter_in_ids.npy')
    dataset = UCDDataset(twitter_in_ids, data_path, 128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for Batch in dataloader:
        print(Batch)

