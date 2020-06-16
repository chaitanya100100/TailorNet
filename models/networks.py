import torch.nn as nn


class FullyConnected(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024, num_layers=None):
        super(FullyConnected, self).__init__()
        net = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        ]
        for i in range(num_layers - 2):
            net.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
            ])
        net.extend([
            nn.Linear(hidden_size, output_size),
        ])
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class FcModified(FullyConnected):
    pass