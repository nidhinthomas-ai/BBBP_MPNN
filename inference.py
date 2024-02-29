import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit.Chem import Draw

from bbbp_mpnn.mpnn import MessagePassing, TransformerEncoderReadout
from bbbp_mpnn.dataset import molecule_from_smiles, graphs_from_smiles, MPNNDataset

def load_model(model_path):
    """
    Load the trained model from the specified path.
    """
    return tf.keras.models.load_model(model_path, custom_objects={'MessagePassing': MessagePassing, 'TransformerEncoderReadout': TransformerEncoderReadout})

def load_test_data(test_dataset_path):
    """
    Load the test dataset.
    """
    df = pd.read_csv(test_dataset_path)
    return df

def predict_properties(model, test_dataset):
    """
    Predict properties of molecules in the test dataset using the trained model.
    """
    y_pred = tf.squeeze(model.predict(test_dataset), axis=1)
    return y_pred

def plot_molecules_in_batches(molecules, y_true, y_pred, output_path, batch_size=16):
    """
    Plot the molecules with their true and predicted properties in batches and save the images.
    """
    num_batches = len(molecules) // batch_size + (len(molecules) % batch_size > 0)
    for batch in range(num_batches):
        start = batch * batch_size
        end = start + batch_size
        batch_molecules = molecules[start:end]
        batch_legends = [f"y_true/y_pred = {y_true[i]}/{y_pred[i]:.2f}" for i in range(start, min(end, len(molecules)))]
        
        fig, axs = plt.subplots(nrows=int(len(batch_molecules)/4) + (len(batch_molecules) % 4 > 0), ncols=4, figsize=(20, 20))
        axs = axs.flatten()
        for ax, molecule, legend in zip(axs, batch_molecules, batch_legends):
            if molecule is not None:
                ax.imshow(Draw.MolToImage(molecule))
                ax.set_title(legend)
            ax.axis('off')
        plt.tight_layout()
        # Save each batch as a separate image
        plt.savefig(f"{output_path}_batch_{batch}.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference with a trained MPNN model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the CSV dataset file.')
    parser.add_argument('--output_path', type=str, default='inference_results', help='Base path to save the output images.')
    args = parser.parse_args()

    model = load_model(args.model_path)
    df = load_test_data(args.dataset_path)

    permuted_indices = np.random.permutation(np.arange(df.shape[0]))
    train_index = permuted_indices[:int(df.shape[0] * 0.8)]
    valid_index = permuted_indices[int(df.shape[0] * 0.8):int(df.shape[0] * 0.99)]
    test_index = permuted_indices[int(df.shape[0] * 0.99):]

    # Featurize and split the data
    x_train, y_train = graphs_from_smiles(df.iloc[train_index]['smiles']), df.iloc[train_index]['p_np']
    x_valid, y_valid = graphs_from_smiles(df.iloc[valid_index]['smiles']), df.iloc[valid_index]['p_np']
    x_test, y_test = graphs_from_smiles(df.iloc[test_index]['smiles']), df.iloc[test_index]['p_np']

    molecules = [molecule_from_smiles(df['smiles'].values[index]) for index in test_index]
    y_true = [df['p_np'].values[index] for index in test_index]

    x_test = graphs_from_smiles([df['smiles'].iloc[index] for index in test_index])
    test_dataset = MPNNDataset(x_test, y_true, batch_size=32)  # Prepare the test dataset

    y_pred = predict_properties(model, test_dataset)

    plot_molecules_in_batches(molecules, y_true, y_pred, args.output_path, batch_size=16)
