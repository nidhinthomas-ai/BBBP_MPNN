import os
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import Dataset, Data
import torch
from bbbp_mpnn.utils import AtomFeaturizer, BondFeaturizer  # Updated for PyTorch

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Define featurizers
atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": {"B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"},
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3"},
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
    }
)

class MoleculeGraphDataset(Dataset):
    def __init__(self, root, file_name, transform=None, pre_transform=None):
        self.file_name = file_name
        super(MoleculeGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.file_name]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass  # Implement this if your dataset needs to be downloaded

    def process(self):
        data_list = []
        df = pd.read_csv(self.raw_paths[0])
        for index, row in df.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            data = self.molecule_to_graph(mol)
            if data:  # Ensure that the molecule could be converted successfully
                data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def molecule_to_graph(self, molecule):
        atom_features = []
        bond_features = []
        edge_index = []

        for atom in molecule.GetAtoms():
            atom_features.append(AtomFeaturizer().featurize(atom))
            # Add self-loops. PyTorch Geometric handles them if necessary
            edge_index.append([atom.GetIdx(), atom.GetIdx()])
            bond_features.append(BondFeaturizer().featurize(None))  # For self-loops

            for neighbor in atom.GetNeighbors():
                bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                edge_index.append([atom.GetIdx(), neighbor.GetIdx()])
                bond_features.append(BondFeaturizer().featurize(bond))

        if not atom_features or not bond_features:
            return None

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        atom_features = torch.stack(atom_features)
        bond_features = torch.stack(bond_features)

        return Data(x=atom_features, edge_attr=bond_features, edge_index=edge_index)
