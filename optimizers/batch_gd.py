import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch

def batch_gd(model, criterion, optimizer, train_loader, val_loader, epochs: int, name: str) -> tuple[list, list, list, list]:
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    train_acces = np.zeros(epochs)
    val_acces = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0

    for iter in tqdm(range(epochs)):
        model.train()
        t0 = datetime.now()
        train_loss = []
        train_acc = []
        for inputs, targets in train_loader:
            # move data to GPU
            inputs, targets = inputs.to(model.device, dtype=torch.float), targets.to(model.device, dtype=torch.int64)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            tmp_acc = torch.count_nonzero(torch.argmax(outputs, dim = 1) == targets).item()/targets.size(0)
            train_acc.append(tmp_acc)
        # Get train loss and test loss
        train_loss = np.mean(train_loss)
        train_acc = np.mean(train_acc)

        model.eval()
        val_loss = []
        val_acc = []
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(model.device, dtype=torch.float), targets.to(model.device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss.append(loss.item())
            tmp_acc = torch.count_nonzero(torch.argmax(outputs, dim=1) == targets).item() / targets.size(0)
            train_acc.append(tmp_acc)
        val_losses = np.mean(val_loss)
        val_acces = np.mean(val_acc)

        # Save losses
        train_losses[iter] = train_loss
        val_losses[iter] = val_loss
        train_acces[iter] = train_acc
        val_losses[iter] = val_acc

        if val_loss < best_test_loss:
            #save_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            #torch.save(model, f'./best_val_model_{name}_{save_time}')
            torch.save(model, f'./best_val_model_{name}')
            best_test_loss = val_loss
            best_test_epoch = iter
            print('model saved')

        dt = datetime.now() - t0
        print(f'Epoch {iter + 1}/{epochs}, \
          Train Loss: {train_loss:.4f}, Train Acc: {train_acc: .4f},\
          Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc: .4f},\
          Duration: {dt}, Best Val Epoch: {best_test_epoch}')

    return train_losses, train_acces, val_losses, val_acces