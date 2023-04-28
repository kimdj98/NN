import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from fast_ml.model_development import train_valid_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

import wandb

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())

# implement custom dataset
class CustomDataset(Dataset):
    def __init__(self, input, label):
        self.input = input
        self.label = label

    def __len__(self):
        return self.input.__len__()

    def __getitem__(self, index):
        return self.input[index], self.label[index]

def load_train_val_test_dataloader(config, device=device):
    # ======================================
    # ---------- Q1 Generate Data ----------
    # ======================================
    x = np.linspace(-10, 10, num=200)
    y = np.linspace(-10, 10, num=200)

    xx, yy = np.meshgrid(x, y)
    zz = -0.0001*(np.abs(np.sin(xx)*np.sin(yy)*np.exp(np.abs(100 - np.sqrt(xx**2 +yy**2)/np.pi))) + 1)**0.1

    # --- data visualization starts

    # fig, ax = plt.subplots(subplot_kw={"projection":"3d"}, figsize = (10,10))
    # surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # plt.show()

    # --- data visualization ends

    # ==========================================
    # ---------- Q2 Add Noise to data ----------
    # ==========================================
    mask = (-10 <= xx) * (xx <= 0) * (-10 <= yy) * (yy <= 0)
    noise = mask * np.random.randn(200, 200)

    zz += noise

    # --- data visualization starts

    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d"}, figsize = (20,20))
    # surf = ax1.plot_surface(xx, yy, noise, cmap=cm.coolwarm)
    # ax1.set_title("Noise")
    # surf = ax2.plot_surface(xx, yy, zz, cmap=cm.coolwarm)
    # ax2.set_title("Noise Generated Data")
    # plt.show()

    # --- data visualization ends

    # ===================================
    # ---------- Q3 Split Data ----------
    # ===================================
    df = pd.DataFrame({"x": xx.reshape(-1), "y": yy.reshape(-1), "z" : zz.reshape(-1)})

    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df, target="z", train_size=0.7, valid_size=0.15, test_size=0.15, method="random")

    # test if y_train + y_valid + y_test is equally distributed (70%, 15%, 15%)
    assert len(y_train) + len(y_valid) + len(y_test) == len(df)
    assert len(df)*0.7 == len(y_train)
    assert (len(df)*0.15) == len(y_valid)
    assert (len(df)*0.15) == len(y_test)

    for data in [X_train, y_train, X_valid, y_valid, X_test, y_test]:
        data = np.array(data)

    # change input type appropriate for the network (DataFrame -> numpy array -> torch tensor(float32))
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32, device=device)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32, device=device)
    X_valid = torch.tensor(np.array(X_valid), dtype=torch.float32, device=device)
    y_valid = torch.tensor(np.array(y_valid), dtype=torch.float32, device=device)
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32, device=device)
    y_test = torch.tensor(np.array(y_test), dtype=torch.float32, device=device)


    # test custom dataset
    train_dataset = CustomDataset(X_train, y_train)
    valid_dataset = CustomDataset(X_valid, y_valid)
    test_dataset = CustomDataset(X_test, y_test)

    batch_size = config["batch_size"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, valid_loader, test_loader


def load_train_test_dataset(config, device=device):
    # ======================================
    # ---------- Q1 Generate Data ----------
    # ======================================
    x = np.linspace(-10, 10, num=200)
    y = np.linspace(-10, 10, num=200)

    xx, yy = np.meshgrid(x, y)
    zz = -0.0001*(np.abs(np.sin(xx)*np.sin(yy)*np.exp(np.abs(100 - np.sqrt(xx**2 +yy**2)/np.pi))) + 1)**0.1

    # --- data visualization starts

    # fig, ax = plt.subplots(subplot_kw={"projection":"3d"}, figsize = (10,10))
    # surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # plt.show()

    # --- data visualization ends

    # ==========================================
    # ---------- Q2 Add Noise to data ----------
    # ==========================================
    mask = (-10 <= xx) * (xx <= 0) * (-10 <= yy) * (yy <= 0)
    noise = mask * np.random.randn(200, 200)

    zz += noise

    # --- data visualization starts

    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d"}, figsize = (20,20))
    # surf = ax1.plot_surface(xx, yy, noise, cmap=cm.coolwarm)
    # ax1.set_title("Noise")
    # surf = ax2.plot_surface(xx, yy, zz, cmap=cm.coolwarm)
    # ax2.set_title("Noise Generated Data")
    # plt.show()

    # --- data visualization ends

    # ===================================
    # ---------- Q3 Split Data ----------
    # ===================================
    df = pd.DataFrame({"x": xx.reshape(-1), "y": yy.reshape(-1), "z" : zz.reshape(-1)})

    X_train, X_test, y_train, y_test = train_test_split(df[["x", "y"]], df["z"], test_size=0.15)

    # test if y_train + y_valid + y_test is equally distributed (70%, 15%, 15%)
    assert len(y_train) + len(y_test) == len(df)
    assert len(df)*0.85 == len(y_train)
    assert (len(df)*0.15) == len(y_test)

    # change input type appropriate for the network (DataFrame -> numpy array -> torch tensor(float32))
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32, device=device)
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32, device=device)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32, device=device)
    y_test = torch.tensor(np.array(y_test), dtype=torch.float32, device=device)

    # test custom dataset
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    return train_dataset, test_dataset

def load_dataset(config, device=device):
    # ======================================
    # ---------- Q1 Generate Data ----------
    # ======================================
    x = np.linspace(-10, 10, num=200)
    y = np.linspace(-10, 10, num=200)

    xx, yy = np.meshgrid(x, y)
    zz = -0.0001*(np.abs(np.sin(xx)*np.sin(yy)*np.exp(np.abs(100 - np.sqrt(xx**2 +yy**2)/np.pi))) + 1)**0.1

    # --- data visualization starts

    # fig, ax = plt.subplots(subplot_kw={"projection":"3d"}, figsize = (10,10))
    # surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # plt.show()

    # --- data visualization ends

    # ==========================================
    # ---------- Q2 Add Noise to data ----------
    # ==========================================
    mask = (-10 <= xx) * (xx <= 0) * (-10 <= yy) * (yy <= 0)
    noise = mask * np.random.randn(200, 200)

    zz += noise

    # --- data visualization starts

    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d"}, figsize = (20,20))
    # surf = ax1.plot_surface(xx, yy, noise, cmap=cm.coolwarm)
    # ax1.set_title("Noise")
    # surf = ax2.plot_surface(xx, yy, zz, cmap=cm.coolwarm)
    # ax2.set_title("Noise Generated Data")
    # plt.show()

    # --- data visualization ends

    # ===================================
    # ---------- Q3 Split Data ----------
    # ===================================
    df = pd.DataFrame({"x": xx.reshape(-1), "y": yy.reshape(-1), "z" : zz.reshape(-1)})

    X, y = df[["x", "y"]], df["z"]

    # test if y_train + y_valid + y_test is equally distributed (70%, 15%, 15%)
    assert len(y) == len(df)

    # change input type appropriate for the network (DataFrame -> numpy array -> torch tensor(float32))
    X = torch.tensor(np.array(X), dtype=torch.float32, device=device)
    y = torch.tensor(np.array(y), dtype=torch.float32, device=device)

    # test custom dataset
    dataset = CustomDataset(X, y)

    return dataset


if __name__== "__main__":
    run_config = {
        'model':'2-8-1',
        'hidden_layer': 8,
        'optimizer':'sgd',
        'lr': 3e-3,
        'batch_size': 2,
        'step_size' : 2,
        'gamma' : 0.8,
        'epochs': 20,
    }

    train_dataset, test_dataset = load_train_test_dataset(run_config)
    pass