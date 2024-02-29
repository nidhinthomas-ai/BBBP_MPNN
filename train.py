import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

from dataset import graphs_from_smiles, MPNNDataset
from model import MPNNModel

def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Train an MPNN model on the BBBP dataset.')
    parser.add_argument('--dataset', type=str, default="BBBP.csv", help='BBBP csv file with the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--message_units', type=int, default=64, help='Number of units in the message passing layer')
    parser.add_argument('--message_steps', type=int, default=4, help='Number of message passing steps')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='Number of attention heads in the transformer encoder')
    parser.add_argument('--dense_units', type=int, default=512, help='Number of units in the dense layer')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train for')
    parser.add_argument('--save_path', type=str, default="./", help='Path to save the model')
    args = parser.parse_args()

    # Reading the dataset into a dataframe
    df = pd.read_csv(args.dataset, usecols=[1, 2, 3]) 

    # Shuffle and split the dataset
    permuted_indices = np.random.permutation(np.arange(df.shape[0]))
    train_index = permuted_indices[:int(df.shape[0] * 0.8)]
    valid_index = permuted_indices[int(df.shape[0] * 0.8):int(df.shape[0] * 0.99)]
    test_index = permuted_indices[int(df.shape[0] * 0.99):]

    # Featurize and split the data
    x_train, y_train = graphs_from_smiles(df.iloc[train_index]['smiles']), df.iloc[train_index]['p_np']
    x_valid, y_valid = graphs_from_smiles(df.iloc[valid_index]['smiles']), df.iloc[valid_index]['p_np']
    x_test, y_test = graphs_from_smiles(df.iloc[test_index]['smiles']), df.iloc[test_index]['p_np']

    # Initialize and compile the MPNN model
    mpnn = MPNNModel(
        atom_dim=x_train[0][0][0].shape[0], bond_dim=x_train[1][0][0].shape[0],
        batch_size=args.batch_size,
        message_units=args.message_units,
        message_steps=args.message_steps,
        num_attention_heads=args.num_attention_heads,
        dense_units=args.dense_units,
    )
    mpnn.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        metrics=[keras.metrics.AUC(name="AUC")],
    )

    # Prepare datasets
    train_dataset = MPNNDataset(x_train, y_train, batch_size=args.batch_size)
    valid_dataset = MPNNDataset(x_valid, y_valid, batch_size=args.batch_size)
    test_dataset = MPNNDataset(x_test, y_test, batch_size=args.batch_size)

    # Define a model checkpoint callback
    model_checkpoint_callback = ModelCheckpoint(
        filepath= args.save_path + "/mpnn_best_model.h5",  # Specify the path where to save the model
        save_best_only=True,  # Set to True to save only the best model according to the validation loss
        monitor='val_loss',  # Specify the metric to monitor
        mode='min',  # The model is saved when the monitored metric has minimized
        verbose=1,
    )

    # Train the model with the checkpoint callback
    history = mpnn.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=args.epochs,
        verbose=2,
        class_weight={0: 2.0, 1: 0.5},
        callbacks=[model_checkpoint_callback],  # Add the callback to the callbacks list
    )

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["AUC"], label="train AUC")
    plt.plot(history.history["val_AUC"], label="valid AUC")
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
