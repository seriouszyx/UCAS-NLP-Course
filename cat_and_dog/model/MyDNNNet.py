from torch import nn, optim


class MyDNNNet(nn.Module):
    def __init__(self):
        super(MyDNNNet, self).__init__()

        self.dnn = nn.Sequential(
            nn.Linear(224 * 224 * 3, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        nn1 = x.view(-1, 224 * 224 * 3)
        # print(nn1.shape)
        out = self.dnn(nn1)
        # print(out.shape)
        return out
