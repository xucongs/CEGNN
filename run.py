import os
import torch
from torch import nn
from pymatgen.core.structure import Structure
from data import CIFData, collate_fn, preprocess_and_save, PreprocessedCIFData
from cgcnn.cgcnn.model import CrystalGraphConvNet
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from model import EGNN
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set the visible CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def read_cif(file_path):
    """Read structure information from a CIF file."""
    return Structure.from_file(file_path)

# Dictionary to store computed adjacency matrices
_edges_dict = {}

def get_adj_matrix(n_nodes, batch_size, device):
    """Get the adjacency matrix for the specified number of nodes and batch size."""
    if n_nodes in _edges_dict:
        edges_dic_b = _edges_dict[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
    
    # Calculate the adjacency matrix
    rows, cols = [], []
    for batch_idx in range(batch_size):
        for i in range(n_nodes):
            for j in range(n_nodes):
                rows.append(i + batch_idx * n_nodes)
                cols.append(j + batch_idx * n_nodes)
    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    _edges_dict.setdefault(n_nodes, {})[batch_size] = edges
    return edges

class EGNNWithRegression(nn.Module):
    def __init__(self, egnn1, egnn2, node_feature_dim, output_dim):
        """Initialize the EGNN with regression model."""
        super(EGNNWithRegression, self).__init__()
        self.egnn_layer1 = egnn1
        self.egnn_layer2 = egnn2
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(node_feature_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, output_dim),
        )
    
    def forward(self, l1, l2, batch_size, n_nodes):
        """Forward pass for the model."""
        out1 = self.egnn_layer1(*l1)
        out2 = self.egnn_layer2(*l2)
        
        # Average pooling of node features
        out1 = out1.view(batch_size, n_nodes, -1).mean(dim=1)
        out2 = out2.view(batch_size, n_nodes, -1).mean(dim=1)
        
        # Concatenate features for regression output
        output = self.fc_layers(torch.cat([out1, out2], dim=-1))
        return output

# Data preprocessing
dataset = CIFData(root_dir='./str/')
save_path = './preprocessed_data_tot.pt'
preprocess_and_save(dataset, save_path)
dataset = PreprocessedCIFData(data_path=save_path)

# K-Fold Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_model():
    """Initialize the EGNN model."""
    egnn = lambda: EGNN(92, 6, 128, device=device, act_fn=nn.SiLU(), n_layers=3, attention=False,
                         norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, 
                         norm_constant=1, inv_sublayers=2, sin_embedding=False, 
                         normalization_factor=100, aggregation_method='sum')
    model = EGNNWithRegression(egnn(), egnn(), node_feature_dim=128, output_dim=1)
    return model.to(device)

def train_and_evaluate(train_idx, val_idx, fold_num=0):
    """Train and evaluate the model."""
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

    model = init_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 200
    best_r2 = 0

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            m1, m2, target = batch
            batch_size = m1[0].size(0)
            n_nodes = m1[0].shape[1]
            edges = get_adj_matrix(n_nodes, batch_size, device=device)

            h1 = m1[0].view(batch_size * n_nodes, -1).clone().to(device)
            h2 = m2[0].view(batch_size * n_nodes, -1).clone().to(device)
            atom_coords1 = m1[1].view(batch_size * n_nodes, -1).clone().to(device)
            atom_coords2 = m2[1].view(batch_size * n_nodes, -1).clone().to(device)
            lattice_matrix1 = m1[2].to(device)
            lattice_matrix2 = m2[2].to(device)
            edge_mask1 = m1[3].view(batch_size * n_nodes * n_nodes, 1).clone().to(device)
            edge_mask2 = m2[3].view(batch_size * n_nodes * n_nodes, 1).clone().to(device)
            node_mask1 = m1[4].view(batch_size * n_nodes, 1).clone().to(device)
            node_mask2 = m2[4].view(batch_size * n_nodes, 1).clone().to(device)

            optimizer.zero_grad()
            layer1 = (h1, atom_coords1, edges, lattice_matrix1, node_mask1, edge_mask1)
            layer2 = (h2, atom_coords2, edges, lattice_matrix2, node_mask2, edge_mask2)

            outputs = model(layer1, layer2, batch_size, n_nodes)
            target = target.to(device)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

        # Evaluate the model
        model.eval()
        val_loss = 0.0
        all_val_outputs, all_val_targets = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                m1, m2, target = batch
                batch_size = m1[0].size(0)
                n_nodes = m1[0].shape[1]
                edges = get_adj_matrix(n_nodes, batch_size, device=device)

                h1 = m1[0].view(batch_size * n_nodes, -1).clone().to(device)
                h2 = m2[0].view(batch_size * n_nodes, -1).clone().to(device)
                atom_coords1 = m1[1].view(batch_size * n_nodes, -1).clone().to(device)
                atom_coords2 = m2[1].view(batch_size * n_nodes, -1).clone().to(device)
                lattice_matrix1 = m1[2].to(device)
                lattice_matrix2 = m2[2].to(device)
                edge_mask1 = m1[3].view(batch_size * n_nodes * n_nodes, 1).clone().to(device)
                edge_mask2 = m2[3].view(batch_size * n_nodes * n_nodes, 1).clone().to(device)
                node_mask1 = m1[4].view(batch_size * n_nodes, 1).clone().to(device)
                node_mask2 = m2[4].view(batch_size * n_nodes, 1).clone().to(device)

                layer1 = (h1, atom_coords1, edges, lattice_matrix1, node_mask1, edge_mask1)
                layer2 = (h2, atom_coords2, edges, lattice_matrix2, node_mask2, edge_mask2)

                outputs = model(layer1, layer2, batch_size, n_nodes)
                target = target.to(device)
                loss = criterion(outputs, target)
                val_loss += loss.item()

                all_val_outputs.append(outputs.cpu().numpy())
                all_val_targets.append(target.cpu().numpy())

        # Calculate average validation loss and R² value
        val_loss /= len(val_dataloader)
        all_val_outputs = np.concatenate(all_val_outputs, axis=0).ravel()
        all_val_targets = np.concatenate(all_val_targets, axis=0).ravel()

        val_r2 = r2_score(all_val_targets, all_val_outputs)
        if best_r2 < val_r2:
            best_r2 = val_r2
            torch.save(model.state_dict(), f'./model_{fold_num}.pt')

        print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}, Validation R²: {val_r2:.4f}')

    return best_r2

# K-Fold Cross-validation training and evaluation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_r2_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f'Fold {fold + 1}/{num_folds}')
    fold_r2 = train_and_evaluate(train_idx, val_idx, fold)
    fold_r2_scores.append(fold_r2)

# Calculate and output the mean R² value
mean_r2 = np.mean(fold_r2_scores)
print(f'Mean R²: {mean_r2:.4f}')

# Testing the model
model = init_model()
model.eval()
test_loss = 0.0

# Load the model state
state_dict = torch.load('./model_1.pt')
model.load_state_dict(state_dict)

# Create the test data loader
test_dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

all_outputs, all_targets = [], []

with torch.no_grad():
    for batch in test_dataloader:
        m1, m2, target = batch
        batch_size = m1[0].size(0)
        n_nodes = m1[0].shape[1]
        edges = get_adj_matrix(n_nodes, batch_size, device=device)

        h1 = m1[0].view(batch_size * n_nodes, -1).clone().to(device)
        h2 = m2[0].view(batch_size * n_nodes, -1).clone().to(device)
        atom_coords1 = m1[1].view(batch_size * n_nodes, -1).clone().to(device)
        atom_coords2 = m2[1].view(batch_size * n_nodes, -1).clone().to(device)
        lattice_matrix1 = m1[2].to(device)
        lattice_matrix2 = m2[2].to(device)
        edge_mask1 = m1[3].view(batch_size * n_nodes * n_nodes, 1).clone().to(device)
        edge_mask2 = m2[3].view(batch_size * n_nodes * n_nodes, 1).clone().to(device)
        node_mask1 = m1[4].view(batch_size * n_nodes, 1).clone().to(device)
        node_mask2 = m2[4].view(batch_size * n_nodes, 1).clone().to(device)

        layer1 = (h1, atom_coords1, edges, lattice_matrix1, node_mask1, edge_mask1)
        layer2 = (h2, atom_coords2, edges, lattice_matrix2, node_mask2, edge_mask2)

        outputs = model(layer1, layer2, batch_size, n_nodes)
        target = target.to(device)
        loss = criterion(outputs, target)
        test_loss += loss.item()

        all_outputs.append(outputs.cpu().numpy())
        all_targets.append(target.cpu().numpy())

# Output test results
test_loss /= len(test_dataloader)
print(f'Test Loss: {test_loss:.4f}')

all_outputs = np.concatenate(all_outputs, axis=0).ravel()
all_targets = np.concatenate(all_targets, axis=0).ravel()

df = pd.DataFrame({"Actual": all_targets, "Predicted": all_outputs})
df.to_csv("predictions.csv", index=False)
print("Data has been written to predictions.csv")

# Calculate and output R² and MSE metrics
r2 = r2_score(all_targets, all_outputs)
mse = mean_squared_error(all_targets, all_outputs)

print(f'R²: {r2:.4f}')
print(f'MSE: {mse:.4f}')

# Create a linear regression plot for predicted results
plt.figure(figsize=(10, 8))
plt.rcParams["font.family"] = "Times New Roman"

sns.scatterplot(x=all_targets, y=all_outputs, s=50, color='blue', alpha=0.6, edgecolor='w', label='Data points')
sns.regplot(x=all_targets, y=all_outputs, scatter=False, color='red', label='Regression line')

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('Actual Band Gap Energy (eV)', fontsize=26)
plt.ylabel('Predicted Band Gap Energy (eV)', fontsize=26)

plt.savefig('predicted_vs_actual.png', dpi=800, bbox_inches='tight')
print("Saved the predicted vs actual comparison image.")
