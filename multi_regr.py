import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchsummary import summary

from autoregressive_dataset import AutoregressiveDataset

from regression_model import RegressionModel
import random

random.seed(42)
torch.manual_seed(42)

def generate_dataset(window_size=500, dataset_size=100000):
    A = torch.rand((window_size, window_size), requires_grad=False)*2.0-1.0
    x = torch.rand((window_size, 1), requires_grad=False)*2.0-1.0

    A = A.cuda()
    x = x.cuda()

    dataset = []
    while len(dataset) < dataset_size:
        x = x / x.norm()
        x1 = A @ x
        x1 = x1[0]
        dataset.append(x1.item())
        x = torch.cat((x[1:], x1.unsqueeze(0)))

    return dataset


def eval_model(model:RegressionModel, test_dataloader:DataLoader):
    model.eval()
    loss_function = nn.MSELoss().cuda()

    total_loss, data_count = 0.0, 0
    for x, y in test_dataloader:
        x = x.cuda()
        y = y.cuda().unsqueeze(1)
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        total_loss += loss.item() * len(x)
        data_count += len(x)

    return total_loss / data_count

def split_ds(dataset, train_split=0.5, val_split=0.1):
    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, train_size + val_size + test_size))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset

def train_model(ds:list[float], train_split=0.5, val_split=0.1, model_size=500, epochs=10, batch_size=4098):
    model = RegressionModel(model_size=model_size).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.5)
    loss_function = nn.MSELoss().cuda()

    dataset = AutoregressiveDataset(window_size=model_size, sequence=ds)
    train_dataset, val_dataset, test_dataset = split_ds(dataset, train_split, val_split)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
                curve_point = (epoch, step, loss.item(), eval_model(model, val_dataloader))
                loss_curve.append(curve_point)
                print(curve_point)

    return loss_curve, eval_model(model, test_dataloader), model


if __name__ == "__main__":
    model_size = 200
    ds = generate_dataset(window_size=500, dataset_size=10000)
    print("dataset stats: ", np.mean(ds), np.std(ds), np.min(ds), np.max(ds))
    print("dataset sample: ", random.sample(ds, 10))
    loss_curve, test_loss, model = train_model(ds, model_size=model_size, epochs=20, batch_size=8192)
    print("params=", sum(p.numel() for p in model.parameters()))
    print("test_loss=",test_loss)


