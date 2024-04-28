import numpy as np

class Featurizer:
    """Base class for featurizing inputs, designed to be extended for specific types of features.

    Attributes:
        dim (int): The dimensionality of the feature vector.
        features_mapping (dict): A mapping from feature names to their indices in the feature vector.
    """
    def __init__(self, allowable_sets):
        """Initializes the Featurizer with allowable sets of features.

        Args:
            allowable_sets (dict): A dictionary where keys are feature names and values are sets of allowable features.
        """
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        """Encodes the inputs into a feature vector.

        Args:
            inputs: The input data to be featurized.

        Returns:
            np.ndarray: A numpy array representing the encoded feature vector.
        """
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    """Featurizer for atoms in molecules, extending the base Featurizer class."""
    def __init__(self, allowable_sets):
        """Initializes the AtomFeaturizer with allowable sets of atom features.

        Args:
            allowable_sets (dict): A dictionary of allowable atom features.
        """
        super().__init__(allowable_sets)

    def symbol(self, atom):
        """Gets the atomic symbol.

        Args:
            atom: An atom object.

        Returns:
            str: The atomic symbol.
        """
        return atom.GetSymbol()

    def n_valence(self, atom):
        """Gets the number of valence electrons.

        Args:
            atom: An atom object.

        Returns:
            int: The number of valence electrons.
        """
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        """Gets the number of hydrogen atoms bonded to the atom.

        Args:
            atom: An atom object.

        Returns:
            int: The number of bonded hydrogen atoms.
        """
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        """Gets the hybridization of the atom.

        Args:
            atom: An atom object.

        Returns:
            str: The hybridization state of the atom, in lowercase.
        """
        return atom.GetHybridization().name.lower()


class BondFeaturizer(Featurizer):
    """Featurizer for bonds in molecules, extending the base Featurizer class."""
    def __init__(self, allowable_sets):
        """Initializes the BondFeaturizer with allowable sets of bond features.

        Args:
            allowable_sets (dict): A dictionary of allowable bond features.
        """
        super().__init__(allowable_sets)
        self.dim += 1  # Adjust dimension for possible None bond representation

    def encode(self, bond):
        """Encodes a bond into a feature vector, with special handling for None (non-existent bonds).

        Args:
            bond: The bond object or None.

        Returns:
            np.ndarray: A numpy array representing the encoded feature vector.
        """
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0  # Mark the last feature as 1 for None bonds
            return output
        return super().encode(bond)

    def bond_type(self, bond):
        """Gets the type of bond.

        Args:
            bond: A bond object.

        Returns:
            str: The type of the bond, in lowercase.
        """
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        """Determines if the bond is conjugated.

        Args:
            bond: A bond object.

        Returns:
            bool: True if the bond is conjugated, False otherwise.
        """
        return bond.GetIsConjugated()
