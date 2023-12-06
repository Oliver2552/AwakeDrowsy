import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

# for training
def train_epoch(model, train_loader, optimizer, loss_function, device):
    model.train()
    running_loss = 0.0
    correct_pred = 0
    total_samples = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
    
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, pred = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_pred += np.sum((pred == labels).cpu().numpy())

    return running_loss / len(train_loader), correct_pred / total_samples

# for evaluating (while training)
def evaluate_epoch(model, valid_loader, loss_function, device):
    model.eval()
    running_loss_val = 0.0
    correct_pred_val = 0
    total_samples_val = 0

    with torch.no_grad():
        for inputs_val, labels_val in valid_loader:
            inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)

            outputs_val = model(inputs_val)
            loss_val = loss_function(outputs_val, labels_val)

            running_loss_val += loss_val.item()

            _, pred_val = torch.max(outputs_val.data, 1)
            total_samples_val += labels_val.size(0)
            correct_pred_val += (pred_val == labels_val).sum().item()

    return running_loss_val / len(valid_loader), correct_pred_val / total_samples_val

# scheduler to change learning rate while training
def adjust_lr(optimizer, scheduler, validation_loss):
    validation_loss = running_loss_val / len(valid_loader)
    scheduler.step(validation_loss)

# current learning rate
def current_learning_rate(optimizer):
    return optimizer.paran_groups[0]['lr']