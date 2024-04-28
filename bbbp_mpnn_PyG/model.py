import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from bbbp_mpnn.mpnn import MessagePassingLayer, TransformerEncoderReadout

class MPNNModel(nn.Module):
    def __init__(self, atom_dim, bond_dim, batch_size=32, message_units=64, message_steps=4, num_attention_heads=8, dense_units=512):
        super(MPNNModel, self).__init__()
        self.message_passing = MessagePassingLayer(atom_dim, bond_dim, message_units, message_steps)
        self.transformer_readout = TransformerEncoderReadout(atom_dim, num_attention_heads, message_units, dense_units, batch_size)
        self.dense1 = nn.Linear(atom_dim, dense_units)  # Adjust input dimension if necessary
        self.dense2 = nn.Linear(dense_units, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.message_passing(x, edge_index, edge_attr)
        x = self.transformer_readout(x, batch)
        x = global_mean_pool(x, batch)  # Pooling to get a single vector for graph representation
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x
