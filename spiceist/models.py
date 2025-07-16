import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GraphEncoder(nn.Module):
    def __init__(self, in_ch, hid_ch, lat_dim, dropout_p=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hid_ch, add_self_loops=False)
        self.bn1   = nn.BatchNorm1d(hid_ch)
        self.dropout1 = nn.Dropout(dropout_p)
        self.conv2 = GCNConv(hid_ch, lat_dim, add_self_loops=False)
        self.bn2   = nn.BatchNorm1d(lat_dim)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(self.bn1(x))
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        x = F.relu(self.bn2(x))
        x = self.dropout2(x)
        return x

class GraphDecoder(nn.Module):
    def __init__(self, lat_dim, hid_ch, out_ch, dropout_p=0.2):
        super().__init__()
        self.fc1    = nn.Linear(lat_dim, hid_ch)
        self.bn1    = nn.BatchNorm1d(hid_ch)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2    = nn.Linear(hid_ch, out_ch)

    def forward(self, z):
        h = self.fc1(z)
        h = F.relu(self.bn1(h))
        h = self.dropout(h)
        return self.fc2(h)

class GraphAutoencoder(nn.Module):
    def __init__(self, in_ch, hid_ch, lat_dim, dropout_p=0.2):
        super().__init__()
        self.encoder = GraphEncoder(in_ch, hid_ch, lat_dim, dropout_p)
        self.decoder = GraphDecoder(lat_dim, hid_ch, in_ch, dropout_p)

    def forward(self, data):
        z_nodes = self.encoder(data.x, data.edge_index)
        x_hat   = self.decoder(z_nodes)
        z_graph = global_mean_pool(z_nodes, data.batch)
        return x_hat, z_graph 