import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch

from loggers import logger


def batch_gd(model_id, model, criterion, optimizer, train_loader, val_loader, epochs, name):
    training_info = {
        'train_loss_hist': [],
        'val_loss_hist': [],
        'train_acc_hist': [],
        'val_acc_hist': []
    }

    best_test_loss = np.inf
    best_test_epoch = 0

    for iter in tqdm(range(epochs)):
        model.train()
        t0 = datetime.now()
        train_loss = []
        train_acc = []
        for inputs, targets in tqdm(train_loader):
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
        for inputs, targets in tqdm(val_loader):
            inputs, targets = inputs.to(model.device, dtype=torch.float), targets.to(model.device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss.append(loss.item())
            tmp_acc = torch.count_nonzero(torch.argmax(outputs, dim=1) == targets).item() / targets.size(0)
            val_acc.append(tmp_acc)
        val_loss = np.mean(val_loss)
        val_acc = np.mean(val_acc)

        # Save losses
        training_info['train_loss_hist'].append(train_loss)
        training_info['val_loss_hist'].append(val_loss)
        training_info['train_acc_hist'].append(train_acc)
        training_info['val_acc_hist'].append(val_acc)

        if val_loss < best_test_loss:
            torch.save(model, os.path.join(logger.find_save_path(model_id), 'best_val_model.pt'))
            best_test_loss = val_loss
            best_test_epoch = iter
            print('model saved')

        dt = datetime.now() - t0
        print(f'Epoch {iter + 1}/{epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc: .4f}, '
              f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc: .4f}, '
              f'Duration: {dt}, Best Val Epoch: {best_test_epoch}')

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
    }, os.path.join(logger.find_save_path(model_id), 'checkpoint.pt'))

    logger.logger(model_id, 'training_info', training_info)
    return