from model import ShallowNet
import torch
import torch.nn as nn
import torch.optim as optim
import preprocessing

import numpy as np

from sklearn.model_selection import KFold

import wandb

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())

def train(config, train_loader, valid_loader):
    # training
    loss_fn = nn.MSELoss(reduction="mean")
    loss_fn2 = nn.MSELoss(reduction="sum")

    epochs = config["epochs"]
    model = ShallowNet(2, config["hidden_layer"], 1)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

    for epoch in range(epochs):
        lr = optimizer.param_groups[0]["lr"]
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


            with torch.no_grad():
                if i % 1000 == 999:
                    last_loss = running_loss / (1000 * config["batch_size"]) # loss per batch
                    print(f'batch {i+1} loss: {last_loss}')
                    wandb.log({"loss": last_loss})
                    running_loss = 0.

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
            
            train_err = np.sqrt(running_train_loss / len(train_loader.sampler.indices))
            print(f'epoch {epoch+1} train_err: {train_err}')
            
            val_err = np.sqrt(running_valid_loss/ len(valid_loader.sampler.indices))
            print(f'epoch {epoch+1} val_err: {val_err}')

            metrics = {"train_eval": train_err,
                        "val_eval": val_err,
                        "lr" : lr
                        }
                
            wandb.log(metrics)

    return model, train_err, val_err

run_config = {
    'model':'base 2-8-1',
    'hidden_layer': 8,
    'optimizer':'sgd',
    'lr': 1e-3,
    'batch_size': 2,
    'step_size' : 2,
    'gamma' : 0.8,
    'epochs': 30,
}

# =============================================
# --------- Q5 hyperparameter tuning ----------
# =============================================

# run = wandb.init(
#     project="NN-Assignment-CV", 
#     job_type="hyper-parameter tuning", 
#     name=run_config['model'],
#     config=run_config
#     )

# train_loader, val_loader, test_loader = preprocessing.load_train_val_test_dataloader(run_config)
# model, train_err, val_err = train(run_config, train_loader, val_loader)

# ==========================================
# --------- Q6 nested CV protocol ----------
# ==========================================
train_dataset, test_dataset = preprocessing.load_train_test_dataset(run_config)

kfold = KFold(3, shuffle=True)

for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
    model_name_base = ""
    model_name_fold = model_name_base + "fold" + str(fold+1) + "_"

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=run_config["batch_size"], sampler=train_subsampler) # 해당하는 index 추출
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=run_config["batch_size"], sampler=val_subsampler)
    
    for hidden_layer_num in [8, 24, 40, 56, 72]:
        model_name = model_name_fold + f"2-{hidden_layer_num}-1"
        run_config["model"] = model_name
        run_config["hidden_layer"] = hidden_layer_num
        run = wandb.init(
            project="NN-Assignment-CV", 
            job_type="Nested CV protocol", 
            name=run_config['model'], 
            config=run_config
            )
        model, train_err, val_err = train(run_config, train_loader, val_loader)
        print(f"{run_config['model']}: {train_err=}, {val_err=}")
        wandb.finish()