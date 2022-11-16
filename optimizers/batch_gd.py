import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch

def batch_gd(device, model, criterion, optimizer, train_loader, val_loader, epochs, name):
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0

    for it in tqdm(range(epochs)):

        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            # move data to GPU
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # Get train loss and test loss
        train_loss = np.mean(train_loss)

        model.eval()
        val_loss = []
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss.append(loss.item())
        val_loss = np.mean(val_loss)

        # Save losses
        train_losses[it] = train_loss
        val_losses[it] = val_loss

        if val_loss < best_test_loss:
            save_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            torch.save(model, f'./best_val_model_{name}_{save_time}')
            best_test_loss = val_loss
            best_test_epoch = it
            print('model saved')

        dt = datetime.now() - t0
        print(f'Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {val_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')

    return train_losses, val_losses