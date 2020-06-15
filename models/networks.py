import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


class Fc(nn.Module):
    def __init__(self, input_size, output_size, num_layers=3, hidden_size=1024):
        super(Fc, self).__init__()
        assert num_layers >= 2
        net = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for i in range(num_layers-2):
            net.append(nn.Linear(hidden_size, hidden_size))
            net.append(nn.ReLU())
        net.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class FcModified(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024, num_layers=None):
        super(FcModified, self).__init__()
        net = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
        ]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class FCBlock(nn.Module):
    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(FCBlock, self).__init__()
        module_list = [nn.Linear(in_size, out_size)]
        if batchnorm:
            module_list.append(nn.BatchNorm1d(out_size))
        if activation is not None:
            module_list.append(activation)
        if dropout:
            module_list.append(dropout)
        self.fc_block = nn.Sequential(*module_list)

    def forward(self, x):
        return self.fc_block(x)
