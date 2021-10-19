from torch import nn


class MyLSTMNet(nn.Module):
    def __init__(self):
        super(MyLSTMNet, self).__init__()
        self.LSTM = nn.LSTM(224 * 3, 128, batch_first=True, num_layers=3)
        self.output = nn.Linear(128, 2)

    def forward(self, x):
        x = x.view(-1, 224, 224 * 3)
        out, (h_n, c_n) = self.LSTM(x)
        return self.output(out[:, -1, :])
