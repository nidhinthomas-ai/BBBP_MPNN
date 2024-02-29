import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolsToGridImage

from utils import AtomFeaturizer, BondFeaturizer

# Suppress TensorFlow logs and warnings for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

def molecule_from_smiles(smiles):
    """
    Convert a SMILES string to an RDKit molecule object, handling sanitization.

    Args:
        smiles (str): SMILES representation of the molecule.

    Returns:
        RDKit Mol object: The molecule represented by the SMILES, with stereochemistry assigned.
    """
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule

def graph_from_molecule(molecule):
    """
    Create a graph representation from an RDKit molecule object.

    Args:
        molecule (RDKit Mol object): The molecule to convert into a graph.

    Returns:
        tuple of numpy arrays: Arrays representing atom features, bond features, and pair indices in the graph.
    """
    atom_features = []
    bond_features = []
    pair_indices = []

    for atom in molecule.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    return np.array(atom_features), np.array(bond_features), np.array(pair_indices)

def graphs_from_smiles(smiles_list):
    """
    Convert a list of SMILES strings to their corresponding graph representations.

    Args:
        smiles_list (list of str): List of SMILES strings.

    Returns:
        tuple of tf.RaggedTensor: Ragged tensors representing atom features, bond features, and pair indices for all molecules.
    """
    atom_features_list = []
    bond_features_list = []
    pair_indices_list = []

    for smiles in smiles_list:
        molecule = molecule_from_smiles(smiles)
        atom_features, bond_features, pair_indices = graph_from_molecule(molecule)

        atom_features_list.append(atom_features)
        bond_features_list.append(bond_features)
        pair_indices_list.append(pair_indices)

    return (
        tf.ragged.constant(atom_features_list, dtype=tf.float32),
        tf.ragged.constant(bond_features_list, dtype=tf.float32),
        tf.ragged.constant(pair_indices_list, dtype=tf.int64),
    )

def prepare_batch(x_batch, y_batch):
    """
    Prepare a batch for training by merging individual graphs into a single, global graph.

    Args:
        x_batch (tuple): Tuple containing atom features, bond features, and pair indices for the batch.
        y_batch (tf.Tensor): The labels for the batch.

    Returns:
        tuple: The merged graph representation and labels for the batch.
    """
    atom_features, bond_features, pair_indices = x_batch
    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()

    molecule_indices = tf.range(len(num_atoms))
    molecule_indicator = tf.repeat(molecule_indices, num_atoms)

    gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
    increment = tf.cumsum(num_atoms[:-1])
    increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices += increment[:, tf.newaxis]
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

    return (atom_features, bond_features, pair_indices, molecule_indicator), y_batch

def MPNNDataset(X, y, batch_size=32, shuffle=False):
    """
    Create a TensorFlow dataset from the provided features and labels.

    Args:
        X (tuple): Features tuple containing atom features, bond features, and pair indices.
        y (np.array): Labels array.
        batch_size (int, optional): Size of the batches. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.

    Returns:
        tf.data.Dataset: The prepared TensorFlow dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(prepare_batch, -1).prefetch(-1)
