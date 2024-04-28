import torch
from torch import nn
from torch_geometric.nn import MessagePassing, GlobalAttention, GATConv
from torch.nn import Linear, Sequential, ReLU, GRUCell, LayerNorm

class EdgeNetwork(MessagePassing):
    def __init__(self, bond_dim, atom_dim):
        super(EdgeNetwork, self).__init__(aggr='add')  # Use 'add' aggregation.
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.lin = Linear(bond_dim, atom_dim * atom_dim)
        self.bias = nn.Parameter(torch.zeros(atom_dim * atom_dim))

    def forward(self, x, edge_index, edge_attr):
        bond_features = self.lin(edge_attr) + self.bias
        bond_features = bond_features.view(-1, self.atom_dim, self.atom_dim)
        return self.propagate(edge_index, x=x, bond_features=bond_features)

    def message(self, x_j, bond_features):
        # x_j denotes the features of source nodes, shape [E, atom_dim]
        transformed_features = torch.matmul(bond_features, x_j.unsqueeze(-1)).squeeze(-1)
        return transformed_features

class MessagePassingLayer(nn.Module):
    def __init__(self, atom_dim, bond_dim, units, steps=4):
        super(MessagePassingLayer, self).__init__()
        self.steps = steps
        self.edge_network = EdgeNetwork(bond_dim, atom_dim)
        self.update_step = GRUCell(atom_dim, atom_dim)

    def forward(self, x, edge_index, edge_attr):
        for _ in range(self.steps):
            m = self.edge_network(x, edge_index, edge_attr)
            x = self.update_step(m, x)
        return x

class TransformerEncoderReadout(nn.Module):
    def __init__(self, atom_dim, num_heads=8, embed_dim=64, dense_dim=512, batch_size=32):
        super(TransformerEncoderReadout, self).__init__()
        self.attention = GATConv(atom_dim, embed_dim, heads=num_heads, concat=False)
        self.dense_proj = Sequential(Linear(embed_dim, dense_dim), ReLU(), Linear(dense_dim, embed_dim))
        self.layernorm1 = LayerNorm(embed_dim)
        self.layernorm2 = LayerNorm(embed_dim)
        self.pooling = GlobalAttention(gate_nn=Linear(embed_dim, 1))

    def forward(self, x, batch):
        x = self.attention(x, batch)
        attention_output = self.attention(x, batch)
        proj_input = self.layernorm1(x + attention_output)
        proj_output = self.layernorm2(proj_input + self.dense_proj(proj_input))
        return self.pooling(proj_output, batch)
