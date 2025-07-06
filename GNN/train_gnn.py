import os
import re
import glob
import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np
import joblib
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import from_networkx

# === CONFIG ===
BASE_PATH = r"C:\Users\user\Desktop\Tesi magistrale Vissicchio\Codice tesi\simulation_output"
ALL_PARAMS = ['beta', 'alpha', 'gamma', 'vacc_prob', 'quar_prob', 'mu']
STATE_NAMES = ['E', 'S', 'V', 'I', 'Q', 'R']

# === Custom Data Class ===
class EpidemicData(Data):
    def __init__(self, x=None, edge_index=None, y=None, global_attr=None):
        super().__init__(x=x, edge_index=edge_index, y=y)
        self.global_attr = global_attr

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'global_attr':
            return 0
        return super().__inc__(key, value, *args, **kwargs)

# === UTILS ===
def extract_metadata_and_data(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    param_line = next((line for line in lines if "Parametri:" in line), None)
    param_dict = eval(re.search(r"\{.*\}", param_line).group(0)) if param_line else {}
    df = pd.read_csv(filepath, comment="#")
    return param_dict, df

def build_graph_from_files(folder):
    # Usa il file NON aggregato, che ha i dati per nodo
    np_csv = glob.glob(os.path.join(folder, "*_np.csv"))[0]
    edgelist_csv = glob.glob(os.path.join(folder, "*_edgelist.csv"))[0]
    param_dict, df = extract_metadata_and_data(np_csv)

    G = nx.read_edgelist(edgelist_csv, delimiter=",", nodetype=int)
    G = nx.convert_node_labels_to_integers(G)

    num_nodes = len(G.nodes)
    feature_matrix = np.zeros((num_nodes, len(STATE_NAMES)))

    # Usa ultimo step per assegnare feature ai nodi
    if 'time' in df.columns and 'node' in df.columns:
        df_last = df[df['time'] == df['time'].max()]
    else:
        raise ValueError("CSV mancante di colonne 'time' o 'node'")

    for node_id in G.nodes:
        row = df_last[df_last['node'] == node_id]
        if not row.empty:
            feature_matrix[node_id] = row[STATE_NAMES].values[0]

    x = torch.tensor(feature_matrix, dtype=torch.float)
    edge_index = from_networkx(G).edge_index
    global_feats = torch.tensor([param_dict.get(k, 0.0) for k in ALL_PARAMS], dtype=torch.float)
    target = torch.tensor(df['I'].max(), dtype=torch.float) if 'I' in df.columns else torch.tensor(0.0)

    return EpidemicData(x=x, edge_index=edge_index, y=target, global_attr=global_feats)


# === DATASET ===
class EpidemicDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.folders = [f for f in glob.glob(os.path.join(root, "**", "sim_*"), recursive=True) if os.path.isdir(f)]
        self.graphs = []
        raw_globals = []
        self.scaler = StandardScaler()

        for folder in tqdm(self.folders, desc="Parsing simulazioni", unit="sim"):
            try:
                data = build_graph_from_files(folder)
                self.graphs.append(data)
                raw_globals.append(data.global_attr.numpy().reshape(1, -1))
            except Exception as e:
                print(f"⚠️ Errore nella cartella {folder}: {e}")

        if not self.graphs:
            raise RuntimeError("Nessun grafo valido trovato.")

        raw_globals = np.vstack(raw_globals)
        self.scaler.fit(raw_globals)
        scaled = self.scaler.transform(raw_globals)
        for i, g in enumerate(self.graphs):
            g.global_attr = torch.tensor(scaled[i], dtype=torch.float)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

# === MODELLO ===
class GNNRegressor(torch.nn.Module):
    def __init__(self, node_feat_dim, global_feat_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim + global_feat_dim, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)

        global_attr = data.global_attr
        if global_attr.dim() == 1:
            global_attr = global_attr.unsqueeze(0)
        if global_attr.size(0) != x.size(0):
            global_attr = global_attr.view(x.size(0), -1)

        x = torch.cat([x, global_attr], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze()

# === TRAINING ===
def train():
    print("📥 Caricamento dataset...")
    dataset = EpidemicDataset(BASE_PATH)
    joblib.dump(dataset.scaler, "scaler.pkl")

    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=16, shuffle=True)
    val_loader = DataLoader([dataset[i] for i in val_idx], batch_size=16)

    model = GNNRegressor(node_feat_dim=dataset[0].x.size(1), global_feat_dim=len(ALL_PARAMS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    print("🚀 Inizio training...")
    for epoch in range(1, 101):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d}", unit="batch", leave=False):
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"📉 Epoch {epoch:03d} - Train Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch)
                val_loss += loss_fn(pred, batch.y).item()
        print(f"          > Val Loss: {val_loss / len(val_loader):.4f}")

    # Crea la cartella 'models' se non esiste
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Salva il modello con la data
    today_str = datetime.now().strftime("%Y-%m-%d")
    model_path = os.path.join(model_dir, f"model_{today_str}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Modello salvato in: {model_path}")

if __name__ == "__main__":
    train()



