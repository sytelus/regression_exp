import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from autoregressive_dataset import AutoregressiveDataset

from regression_model import RegressionModel

torch.manual_seed(42)

def generate_dataset(window_size=500, dataset_size=100000, train_split=0.5):
    A = torch.rand((window_size, window_size), requires_grad=False)*2.0-1.0
    x = torch.rand((window_size, 1), requires_grad=False)*2.0-1.0

    dataset = []
    while len(dataset) < dataset_size:
        x = x / x.norm()
        x1 = A @ x
        x1 = x1[0]
        dataset.append(x1.item())
        x = torch.cat((x[1:], x1.unsqueeze(0)))

    train_size = int(dataset_size * train_split)
    test_size = len(dataset) - train_size

    return dataset[:train_size], dataset[test_size:]


def eval_model(model:RegressionModel, test_data:list[float], batch_size):
    model.eval()
    test_dataset = AutoregressiveDataset(window_size=model.model_size, sequence=test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    loss_function = nn.MSELoss().cuda()

    total_loss = 0.0
    for x, y in test_dataloader:
        x = x.cuda()
        y = y.cuda().unsqueeze(1)
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        total_loss += loss.item() * len(x)

    return total_loss / len(test_dataset)

def train_model(train_data:list[float], test_data:list[float], model_size=500, epochs=10, batch_size=4098):
    model = RegressionModel(model_size=model_size).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_function = nn.MSELoss().cuda()
    train_dataset = AutoregressiveDataset(window_size=model_size, sequence=train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    loss_curve = []
    step = 0
    for epoch in range(epochs):
        for x, y in train_dataloader:
            model.train()

            x = x.cuda()
            y = y.cuda().unsqueeze(1)
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            if step % 1000:
                curve_point = (epoch, step, loss.item()/len(x), eval_model(model, test_data, batch_size))
                loss_curve.append(curve_point)
                print(curve_point)

    return loss_curve


if __name__ == "__main__":
    train_data, test_data = generate_dataset(window_size=500, dataset_size=100000)
    train_model(train_data, test_data, model_size=500, epochs=20, batch_size=8192)


