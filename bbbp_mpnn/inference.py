import argparse
import tensorflow as tf
from rdkit.Chem.Draw import MolsToGridImage
import pandas as pd
from tensorflow.keras.preprocessing.image import save_img

from utils import molecule_from_smiles
from dataset import graphs_from_smiles, MPNNDataset

def load_model(model_path):
    """
    Load the trained model from the specified path.
    """
    return tf.keras.models.load_model(model_path)

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

def plot_results(molecules, y_true, y_pred, output_path):
    """
    Plot the molecules with their true and predicted properties and save the image.
    """
    legends = [f"y_true/y_pred = {y_true[i]}/{y_pred[i]:.2f}" for i in range(len(y_true))]
    img = MolsToGridImage(molecules, molsPerRow=4, legends=legends)
    save_img(output_path, img)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference with a trained MPNN model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the CSV dataset file.')
    parser.add_argument('--output_path', type=str, default='inference_results.png', help='Path to save the output image.')
    args = parser.parse_args()

    model = load_model(args.model_path)
    df = load_test_data(args.dataset_path)

    # You need to adjust this part to prepare the test_dataset from df
    # Assuming you have a mechanism to identify test entries or you use the entire dataset for inference
    test_index = range(len(df))  # Example to use the entire dataset
    molecules = [molecule_from_smiles(df['smiles'].values[index]) for index in test_index]
    y_true = [df['p_np'].values[index] for index in test_index]
    
    # Prepare the test dataset for prediction
    # This step will vary based on your dataset preparation in the training phase
    x_test = graphs_from_smiles([df['smiles'].iloc[index] for index in test_index])
    test_dataset = MPNNDataset(x_test, batch_size=32)  # Assuming batch_size used during training

    y_pred = predict_properties(model, test_dataset)
    
    plot_results(molecules, y_true, y_pred, args.output_path)
