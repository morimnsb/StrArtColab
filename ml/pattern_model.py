import torch, torch.nn as nn

class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Flatten(),
            nn.Linear(64*8*8,128), nn.ReLU(),
            nn.Linear(128,1), nn.Sigmoid()
        )
    def forward(self,x): return self.net(x).squeeze(1)
