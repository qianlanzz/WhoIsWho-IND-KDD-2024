import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch


class MyGCNModel(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(MyGCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.dropout(x, training=self.training)

        x = self.fc(x)
        return torch.sigmoid(x)
