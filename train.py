import os
import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# --- Dataset Loader ---
class ASVspoofDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sr = torchaudio.load(file_path)
        mel = torchaudio.transforms.MelSpectrogram(sr)(waveform)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
        label = self.labels[idx]
        return mel_db.squeeze(0), label

# --- Protocol Parser ---
def parse_protocol_file(protocol_path, audio_base_dir):
    file_names = []
    labels = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[1] + ".flac"
            label = 1 if parts[-1] == "spoof" else 0
            file_names.append(os.path.join(audio_base_dir, filename))
            labels.append(label)
    return file_names, labels

# --- LCNN + optional FTANet ---
class MFM(nn.Module):
    def __init__(self, in_features, out_features, type=0):
        super(MFM, self).__init__()
        self.out_features = out_features
        if type == 0:
            self.filter = nn.Linear(in_features, out_features * 2)
        else:
            self.filter = nn.Conv2d(in_features, out_features * 2, kernel_size=3, padding=1)
        self.type = type

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_features, 1)
        return torch.max(out[0], out[1])

class LCNN(nn.Module):
    def __init__(self, attention=False):
        super(LCNN, self).__init__()
        self.attention = attention

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            MFM(64, 32, type=1),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            MFM(64, 32, type=1),
            nn.MaxPool2d(2)
        )

        self.attn_time = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.Sigmoid()
        ) if attention else None

        self.fc1 = nn.Linear(32 * 24 * 24, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)

        if self.attention:
            attn_mask = self.attn_time(x)
            x = x * attn_mask

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)).squeeze(1)
        return x

# --- Training & Evaluation ---
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    for X, y in tqdm(dataloader, desc="Training"):
        X, y = X.to(device), y.float().to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Evaluating"):
            X = X.to(device)
            outputs = model(X).cpu().numpy()
            all_preds.extend(outputs)
            all_labels.extend(y.numpy())
    auc = roc_auc_score(all_labels, all_preds)
    return auc

# --- Main Training Loop ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modify paths here
    train_protocol = "data/ASVspoof2019.LA.cm.train.trn.txt"
    dev_protocol = "data/ASVspoof2019.LA.cm.dev.trl.txt"
    train_audio_dir = "data/train"
    dev_audio_dir = "data/dev"

    train_files, train_labels = parse_protocol_file(train_protocol, train_audio_dir)
    dev_files, dev_labels = parse_protocol_file(dev_protocol, dev_audio_dir)

    train_set = ASVspoofDataset(train_files, train_labels)
    dev_set = ASVspoofDataset(dev_files, dev_labels)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(dev_set, batch_size=8)

    model = LCNN(attention=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(5):
        print(f"\nEpoch {epoch+1}")
        loss = train(model, train_loader, optimizer, criterion, device)
        auc = evaluate(model, val_loader, device)
        print(f"Loss: {loss:.4f} | AUC: {auc:.4f}")

if __name__ == "__main__":
    main()

