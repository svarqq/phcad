import torch
import math
import time

from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt

input_dim = 2
hidden_dim = 10
output_dim = 1


# Create a dataset with 10,000 samples.
X, y = make_circles(n_samples=10000, noise=0.05, random_state=26)
print(X.shape)


class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


train_data = Data(X, y)
batch_size = 64
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.sigmoid(self.layer_2(x))

        return x


model = NeuralNetwork(input_dim, hidden_dim, output_dim)
print(model)

num_epochs = 100
loss_values = []
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return (
        1
        / (2 * math.pi * sx * sy)
        * torch.exp(-((x - mx) ** 2 / (2 * sx**2) + (y - my) ** 2 / (2 * sy**2)))
    )


def preloss(x):
    x0, y0 = torch.randint(x.shape[0], (2, 20))
    origins = torch.stack((x0, y0)).T
    a = torch.zeros((64, 64))
    print(a.shape)
    for x0, y0 in origins:
        a += gaussian_2d(
            torch.linspace(0, 64, 64), y=torch.linspace(0, 64, 64), mx=x0, my=y0
        )
        plt.pcolormesh(range(64), range(64), a.detach().numpy())
        plt.show()
        time.sleep(2**8)
    print(torch.sum(a))
    return a


def loss(x, _):
    x2 = preloss(x)
    return torch.sum(x2)


loss_fn = loss

for epoch in range(num_epochs):
    for X, y in train_dataloader:
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(-1))
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()

print("Training Complete")
