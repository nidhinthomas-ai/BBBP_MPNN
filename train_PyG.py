import argparse
import torch
import pandas as pd
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.model_selection import train_test_split
from bbbp_mpnn.model import MPNNModel  # Make sure this imports your PyTorch Geometric model
from bbbp_mpnn.dataset import MoleculeGraphDataset  # Adjust this to your dataset class

def train(epoch, model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y.view(-1, 1).float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            output = model(data)
            loss = criterion(output, data.y.view(-1, 1).float())
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser(description='Train an MPNN model on the BBBP dataset.')
    parser.add_argument('--dataset', type=str, default="BBBP.csv", help='BBBP csv file with the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for the optimizer')
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and prepare the dataset
    df = pd.read_csv(args.dataset)
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
    df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=42)

    train_dataset = MoleculeGraphDataset(df_train)
    val_dataset = MoleculeGraphDataset(df_val)
    test_dataset = MoleculeGraphDataset(df_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model
    model = MPNNModel(atom_dim=..., bond_dim=..., message_units=64, message_steps=4, num_attention_heads=8, dense_units=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCELoss()

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch, model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'mpnn_model.pth')

if __name__ == "__main__":
    main()
