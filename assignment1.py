import torch
import torch.nn as nn
import torch.optim as optim

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

# ======================================
# ---------- Q1 Generate Data ----------
# ======================================
x = np.linspace(-10, 10, num=200)
y = np.linspace(-10, 10, num=200)

xx, yy = np.meshgrid(x, y)
zz = -0.0001*(np.abs(np.sin(xx)*np.sin(yy)*np.exp(np.abs(100 - np.sqrt(xx**2 +yy**2)/np.pi))) + 1)**0.1

# --- data visualization starts

fig, ax = plt.subplots(subplot_kw={"projection":"3d"}, figsize = (10,10))
surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()

# --- data visualization ends

# ==========================================
# ---------- Q2 Add Noise to data ----------
# ==========================================
mask = (-10 <= xx) * (xx <= 0) * (-10 <= yy) * (yy <= 0)
noise = mask * np.random.randn(200, 200)

zz += noise

# --- data visualization starts

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d"}, figsize = (20,20))
surf = ax1.plot_surface(xx, yy, noise, cmap=cm.coolwarm)
ax1.set_title("Noise")
surf = ax2.plot_surface(xx, yy, zz, cmap=cm.coolwarm)
ax2.set_title("Noise Generated Data")
plt.show()

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

# ============================================================
# ---------- Q4 Implement Single Hidden Layer Model ----------
# ============================================================
class ShallowNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ShallowNet, self).__init__()
        self.MLP = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   nn.Tanh(),
                                   nn.Linear(hidden_size, output_size))
        
        self.apply(self._init_weights) # weight initialization

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # nn.init.xavier_uniform_(module.weight.data)
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.MLP(x)

model = ShallowNet(2, 8, 1)
model.to(device)
assert model.MLP[0].weight.device == X_train.device, "Model and Data is running on different device"

# ===============================================
# ---------- Q5 Implement LR scheduler ----------
# ===============================================
run_config = {
    'optimizer':'sgd',
    'lr': 3e-3,
    'batch_size': 2,
    'step_size' : 2,
    'gamma' : 0.8,
    'epochs': 20,
}

optimizer = torch.optim.SGD(model.parameters(), lr=run_config["lr"])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=run_config["step_size"], gamma=run_config["gamma"])


# ---------- Implement DataLoader ----------
from torch.utils.data import Dataset, DataLoader

# implement custom dataset
class CustomDataset(Dataset):
    def __init__(self, input, label):
        self.input = input
        self.label = label

    def __len__(self):
        return self.input.__len__()

    def __getitem__(self, index):
        return self.input[index], self.label[index]

# test custom dataset
train_dataset = CustomDataset(X_train, y_train)
valid_dataset = CustomDataset(X_valid, y_valid)
test_dataset = CustomDataset(X_test, y_test)

batch_size = run_config["batch_size"]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# ---------- Train model ----------
loss_fn = nn.MSELoss(reduction="mean")
loss_fn2 = nn.MSELoss(reduction="sum")

def train_epoch():
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        inputs, labels = data

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        running_loss += loss.item()

        wandb.log({"loss": loss})

        with torch.no_grad():
            if i % 1000 == 999:
                last_loss = running_loss / (1000 * batch_size) # loss per batch
                print(f'batch {i+1} loss: {last_loss}')
                running_loss = 0.
    
    return model

def train(config = run_config):
    epochs = config["epochs"]
    model = ShallowNet(2, config["hidden_layer"], 1)
    model.to(device)

    for epoch in range(epochs):
        lr = optimizer.param_groups[0]["lr"]
        model = train_epoch()
        scheduler.step()
        
        with torch.no_grad():
            # evaluate valid and train set
            running_valid_loss = 0.                
            for j, data in enumerate(valid_loader):
                inputs, labels = data
                outputs = model(inputs)
                loss = loss_fn2(outputs, labels)
                running_valid_loss += loss.item()

            running_train_loss = 0.
            for j, data in enumerate(train_loader):
                inputs, labels = data
                outputs = model(inputs)
                loss = loss_fn2(outputs, labels)
                running_train_loss += loss.item()
            
            train_err = np.sqrt(running_train_loss / len(train_dataset))
            print(f'epoch {epoch+1} train_err: {train_err}')
            
            val_err = np.sqrt(running_valid_loss/ len(valid_dataset))
            print(f'epoch {epoch+1} val_err: {val_err}')

            metrics = {"train_eval": train_err,
                        "val_eval": val_err,
                        "lr" : lr
                        }
                
            wandb.log(metrics)

    return model, loss

# run_config['model'] = '2-8-1'
# run_config['hidden_layer'] = 8
# run = wandb.init(
#     project="NN-Assignment1", 
#     job_type="hyper-parameter tuning", 
#     name=run_config['model'], 
#     config=run_config
#     )
# train(run_config)

# ================================================================
# ---------- Q6 Nested 3-Fold Cross Validation Protocol ----------
# ================================================================
# run_config['model'] = '2-20-1'
# run_config['hidden_layer'] = 20
# run = wandb.init(
#     project="NN-Assignment1", 
#     job_type="cross-validation", 
#     name=run_config['model'], 
#     config=run_config
#     )
# train(run_config)

# run_config['model'] = '2-40-1'
# run_config['hidden_layer'] = 40
# run = wandb.init(
#     project="NN-Assignment1", 
#     job_type="cross-validation", 
#     name=run_config['model'], 
#     config=run_config
#     )
# train(run_config)

# run_config['model'] = '2-56-1'
# run_config['hidden_layer'] = 56
# run = wandb.init(
#     project="NN-Assignment1", 
#     job_type="cross-validation", 
#     name=run_config['model'], 
#     config=run_config
#     )
# train(run_config)

# run_config['model'] = '2-72-1'
# run_config['hidden_layer'] = 72
# run = wandb.init(
#     project="NN-Assignment1", 
#     job_type="cross-validation", 
#     name=run_config['model'], 
#     config=run_config
#     )
# train(run_config)

# pass