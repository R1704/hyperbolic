import torch
import torch.nn.functional as F
from models import GCN

class Trainer:
    def __init__(self, dataset):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GCN(dataset).to(device)
        self.data = dataset[0].to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)

    def train_gca(self):
        self.model.train()
        for epoch in range(200):
            self.optimizer.zero_grad()
            out = self.model(self.data)
            loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        pred = self.model(self.data).argmax(dim=1)
        correct = (pred[self.data.test_mask] == self.data.y[self.data.test_mask]).sum()
        acc = int(correct) / int(self.data.test_mask.sum())
        print(f'Accuracy: {acc:.4f}')
