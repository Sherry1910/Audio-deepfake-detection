import torch
import torch.nn as nn

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
        x = x.unsqueeze(1)  # [B, 1, F, T]
        x = self.layer1(x)
        x = self.layer2(x)

        if self.attention:
            attn_mask = self.attn_time(x)
            x = x * attn_mask

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)).squeeze(1)
        return x

