import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from mpnn import MessagePassing, TransformerEncoderReadout

def MPNNModel(
    atom_dim,
    bond_dim,
    batch_size=32,
    message_units=64,
    message_steps=4,
    num_attention_heads=8,
    dense_units=512,
):
    """
    Constructs a Message Passing Neural Network (MPNN) model for molecular property prediction.
    
    Args:
        atom_dim (int): Dimension of the atom feature vector.
        bond_dim (int): Dimension of the bond feature vector.
        batch_size (int): Batch size for the model.
        message_units (int): Number of units in the message passing layers.
        message_steps (int): Number of message passing steps.
        num_attention_heads (int): Number of attention heads in the transformer encoder.
        dense_units (int): Number of units in the dense layer following the transformer encoder.
    
    Returns:
        keras.Model: A Keras model object representing the MPNN.
    """
    # Define input layers for atom features, bond features, pair indices, and molecule indicator
    atom_features = layers.Input(shape=(atom_dim,), dtype="float32", name="atom_features")
    bond_features = layers.Input(shape=(bond_dim,), dtype="float32", name="bond_features")
    pair_indices = layers.Input(shape=(2,), dtype="int32", name="pair_indices")
    molecule_indicator = layers.Input(shape=(), dtype="int32", name="molecule_indicator")

    # Apply message passing with the specified number of units and steps
    x = MessagePassing(message_units, message_steps)([atom_features, bond_features, pair_indices])

    # Apply transformer encoder readout layer
    x = TransformerEncoderReadout(num_attention_heads, message_units, dense_units, batch_size)([x, molecule_indicator])

    # Add dense layers for prediction
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    # Construct and return the model
    model = keras.Model(inputs=[atom_features, bond_features, pair_indices, molecule_indicator], outputs=[x])
    return model
