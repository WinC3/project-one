import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch

import data_parser as dp

import matplotlib.pyplot as plt


class RegressionNN(nn.Module):
    def __init__(self, n_layers, n_input_bins=88, n_notes=88):
        super().__init__()

        n_intermediate_layers = n_layers - 2
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_input_bins, 512))
        for i in range(n_intermediate_layers):
            self.layers.append(nn.Linear(512, 512))
        self.layers.append(nn.Linear(512, n_notes))

    def get_weight_norm(self):
        return sum(torch.norm(m.weight, 2) ** 2
                   for m in self.modules()
                   if isinstance(m, nn.Linear))

    def forward(self, inputs):
        
        out = inputs
        for layer in self.layers[:-1]:
            out = torch.relu(layer(out))
        #out = torch.sigmoid(self.layers[-1](out))
        out = self.layers[-1](out)

        return out


class PitchCNN(nn.Module):
    def __init__(self, num_notes=88, input_shape=(1, 256, 128)):
        """
        Args:
            num_notes: number of piano notes (default=88: A0–C8)
            input_shape: (channels, freq_bins, time_frames)
                         e.g. (1, 256, 128) for spectrogram input
        """
        super().__init__()
        
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(51,5), padding=(25, 2))   # [B, 32, F, T]
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(15,3), padding=(7, 1))  # [B, 64, F, T]
        #self.conv3 = nn.Conv2d(64, 128, kernel_size=(8,2), padding=1) # [B, 128, F, T]
        self.pool = nn.MaxPool2d(2, 2)

        self.residual_conv = nn.Conv2d(1, 16, kernel_size=1)

        # figure out the flattened size automatically
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)  # [B=1, C=1, F, T]
            print("Input shape:", dummy.shape)
            dummy = self._forward_conv(dummy)
            flat_size = dummy.shape[1] * dummy.shape[2] * dummy.shape[3]

        # Fully connected layers
        self.fc1 = nn.Linear(flat_size, 512)
        self.fc2 = nn.Linear(512, num_notes)
        #self.fc3 = nn.Linear(512, 512)
        #self.fc4 = nn.Linear(512, num_notes)


    def _forward_conv(self, x):
        """Pass through conv/pool layers only (for shape calc + reuse)"""
        residual = x
        
        #x = self.pool(F.relu(self.conv1(x)))   # [B, 32, F/2, T/2]
        x = F.relu(self.conv1(x))
        #x = self.pool(F.relu(self.conv2(x)))   # [B, 64, F/4, T/4]
        x = F.relu(self.conv2(x))
        #x = self.pool(F.relu(self.conv3(x)))   # [B, 128, F/8, T/8]

        # Residual connection: transform input to match output dimensions
        if residual.shape[1] != x.shape[1]:  # If channel dimensions don't match
            residual = self.residual_conv(residual)

        return x + residual

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # flatten all dims except batch
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = self.fc2(x)
        x = torch.sigmoid(self.fc2(x))  # [B, num_notes], values in (0,1)
        return x

def visualize_feature_maps(model, sample_input):
    model.eval()
    hooks = []
    feature_maps = []
    
    # Hook to capture conv outputs
    def hook_fn(module, input, output):
        feature_maps.append(output.detach())
    
    # Register hooks on all conv layers
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        _ = model(sample_input.unsqueeze(0))
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Visualize
    for i, fm in enumerate(feature_maps):
        print(f"Layer {i+1} feature maps: {fm.shape}")
        # Show first few channels
        fig, axes = plt.subplots(4, 8, figsize=(12, 6))
        for j, ax in enumerate(axes.flat):
            if j < min(32, fm.shape[1]):
                ax.imshow(fm[0, j].cpu(), cmap='hot')
                ax.set_title(f'FM {j}')
                ax.axis('off')
        plt.suptitle(f'Layer {i+1} Feature Maps')
        plt.show()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)  # pt = probability of correct classification
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss

class FocalLossWithSigmoid(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        # inputs are probabilities (after sigmoid)
        # targets are the same shape as inputs
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1-inputs)  # probability of true class
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()

class WeightedMSELoss(nn.Module):
    def __init__(self, note_weights=None, active_boost=1.0):
        """
        note_weights: Tensor of shape (num_notes,) giving a weight per note
        active_boost: multiplier for loss on active notes (labels > 0)
        """
        super(WeightedMSELoss, self).__init__()
        self.note_weights = note_weights
        self.active_boost = active_boost

    def forward(self, preds, targets):
        """
        preds:   (batch_size, num_notes) predicted volumes
        targets: (batch_size, num_notes) true volumes
        """
        errors = (preds - targets) ** 2  # (batch, notes)

        # Weight errors by note frequency (if provided)
        if self.note_weights is not None:
            errors = errors * self.note_weights.unsqueeze(0)  # broadcast to batch

        # Optionally boost errors for active notes
        if self.active_boost != 1.0:
            active_mask = (targets > 0).float()  # (batch, notes)
            errors = errors * (1.0 + self.active_boost * active_mask)

        # Mean over batch and notes
        return errors.mean()


def train(model, lr, lamb, train_data, train_labels, valid_data, valid_labels, 
          num_epoch, batch_size=128):
    
    train_loader, valid_dataset, optimizer, loss_func = CNN_datasets(model, lr, train_data, train_labels, valid_data, valid_labels, 
                                                   batch_size)

    loss_func = nn.BCELoss()
    #note_weights = calculate_note_weights(train_labels)
    #loss_func = nn.BCEWithLogitsLoss(weight=note_weights)
    #positive_weight = (train_labels == 0).sum() / (train_labels == 1).sum()
    #loss_func = FocalLoss(alpha=positive_weight, gamma=2.0)
    #positive_weight = (train_labels == 0).sum() / (train_labels == 1).sum()
    #loss_func = FocalLossWithSigmoid(alpha=positive_weight, gamma=3.0)
    #positive_weight = (train_labels == 1).sum() / (train_labels == 0).sum()
    #loss_func = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    train_losses = []
    val_accuracies = []

    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            if hasattr(model, 'get_weight_norm') and lamb > 0:
                # Add L2 regularization
                loss += (lamb / 2) * model.get_weight_norm()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)  # rescale by batch for total loss

        avg_train_loss = train_loss / len(train_loader.dataset)

        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                outputs = model(inputs)
                preds = (outputs >= 0.5).float()
                correct += (preds == targets).sum().item()
                total += targets.numel()

            val_acc = correct / total
            train_losses.append(avg_train_loss)
            val_accuracies.append(val_acc)

            val_pred = model(valid_data)
            #metrics = calculate_metrics(valid_labels, val_pred)
            metrics = comprehensive_metrics(valid_labels, val_pred, threshold=0.5)

            print(f"Epoch {epoch+1}/{num_epoch}, "
                f"Train Loss: {avg_train_loss:.10f}, Val Acc: {val_acc:.4f}")
            print(f"  MSE: {metrics['mse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  R2: {metrics['r2']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  Explained Variance: {metrics['explained_variance']:.4f}")
            #print(f"  Accuracy: {metrics['accuracy']:.4f} (mostly useless)")
            #print(f"  Precision: {metrics['precision']:.4f} - note prediction accuracy")
            #print(f"  Recall: {metrics['recall']:.4f} - note detection rate")  
            #print(f"  F1: {metrics['f1']:.4f} - overall balance")

    return train_losses, val_accuracies
    

def classicNN_datasets(model, lr, train_data, train_labels, valid_data, valid_labels, 
                        batch_size=128):

    # datasets
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TensorDataset(valid_data, valid_labels)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    return train_loader, valid_dataset, optimizer, loss_func

def CNN_datasets(model, lr, train_data, train_labels, valid_data, valid_labels, 
                        batch_size=128):

    # datasets
    train_dataset = TensorDataset(train_data, train_labels)  # add channel dim
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TensorDataset(valid_data, valid_labels)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.BCELoss()

    return train_loader, valid_dataset, optimizer, loss_func


def calculate_note_weights(y_train):
    """
    Calculate class weights for each of the 88 piano notes
    Returns: Tensor of shape (88,) with weight for each note
    """
    # Calculate note frequencies
    note_frequencies = y_train.mean(dim=0)  # Shape: (88,)
    
    # Avoid division by zero
    note_frequencies = torch.clamp(note_frequencies, min=1e-8, max=1.0)
    
    # Calculate weights
    note_weights = 1.0 / note_frequencies
    
    # Normalize
    note_weights = note_weights / note_weights.mean()
    
    # Detach and clone to avoid warning
    return note_weights.detach().clone().requires_grad_(False)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, accuracy_score, precision_score, recall_score, f1_score


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculate meaningful metrics for imbalanced classification
    """
    # Convert to binary predictions
    y_pred_binary = (y_pred >= threshold).float()
    
    # Flatten to 1D arrays
    y_true_flat = y_true.cpu().numpy().flatten()
    y_pred_flat = y_pred_binary.cpu().numpy().flatten()
    
    # Calculate metrics
    metrics = {
        'accuracy': (y_pred_binary == y_true).float().mean().item(),
        'precision': precision_score(y_true_flat, y_pred_flat, zero_division=0),
        'recall': recall_score(y_true_flat, y_pred_flat, zero_division=0),
        'f1': f1_score(y_true_flat, y_pred_flat, zero_division=0)
    }
    
    return metrics

def regression_metrics(y_true, y_pred):
    """
    Metrics for continuous velocity prediction
    """
    # Convert to numpy for sklearn
    y_true = y_true.cpu().numpy().flatten()
    y_pred = y_pred.cpu().numpy().flatten()
    
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred)
    }
    
    return metrics

def note_detection_metrics(y_true, y_pred, threshold=0.1):
    """
    Also track note detection performance by thresholding
    """
    y_true_binary = (y_true > threshold).float()
    y_pred_binary = (y_pred > threshold).float()
    
    y_true_flat = y_true_binary.cpu().numpy().flatten()
    y_pred_flat = y_pred_binary.cpu().numpy().flatten()
    
    return {
        'accuracy': accuracy_score(y_true_flat, y_pred_flat),
        'precision': precision_score(y_true_flat, y_pred_flat, zero_division=0),
        'recall': recall_score(y_true_flat, y_pred_flat, zero_division=0),
        'f1': f1_score(y_true_flat, y_pred_flat, zero_division=0)
    }

def comprehensive_metrics(y_true, y_pred, threshold=0.5):
    """
    Track both regression and classification performance
    """
    reg_metrics = regression_metrics(y_true, y_pred)
    cls_metrics = note_detection_metrics(y_true, y_pred, threshold)
    
    return {**reg_metrics, **cls_metrics}

def regression_metrics_by_note(y_true, y_pred):
    """
    Calculate regression metrics for each note separately
    """
    metrics_by_note = {}
    for note in range(y_true.shape[1]):
        metrics_by_note[note] = regression_metrics(y_true[:, note], y_pred[:, note])
    return metrics_by_note

def regression_metrics_by_class(y_true, y_pred):
    """
    Calculate regression metrics for when note is active vs inactive
    """
    metrics_by_class = {'active': {}, 'inactive': {}}
    for note in range(y_true.shape[1]):
        active_mask = y_true[:, note] > 0
        inactive_mask = y_true[:, note] == 0
        
        if active_mask.sum() > 0:
            metrics_by_class['active'][note] = regression_metrics(y_true[active_mask, note], y_pred[active_mask, note])
        else:
            metrics_by_class['active'][note] = None
        
        if inactive_mask.sum() > 0:
            metrics_by_class['inactive'][note] = regression_metrics(y_true[inactive_mask, note], y_pred[inactive_mask, note])
        else:
            metrics_by_class['inactive'][note] = None
            
    return metrics_by_class

def print_note_and_class_metrics(y_true, y_pred):
    """
    Print regression metrics by note and by active/inactive class
    """
    with torch.no_grad():
        note_metrics = regression_metrics_by_note(y_true, y_pred)
        class_metrics = regression_metrics_by_class(y_true, y_pred)
    
    print("Regression Metrics by Note:")
    for note, metrics in note_metrics.items():
        print(f"Note {note}: {metrics}")
    
    print("\nRegression Metrics by Class:")
    for cls, metrics_dict in class_metrics.items():
        print(f"\nClass: {cls}")
        for note, metrics in metrics_dict.items():
            print(f"Note {note}: {metrics}")


def ensure_tensor(data, dtype=torch.float32):
    """Convert data to tensor if it isn't already"""
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).type(dtype)
    else:
        return torch.tensor(data, dtype=dtype)


def evaluate(model, data, labels, batch_size=None, threshold=0.5):
    model.eval()
    
    if batch_size is None:
        batch_size = 128  # Set a reasonable default
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        # Process in batches
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            outputs = model(batch_data)
            predictions = (outputs >= threshold).float()
            correct = (predictions == batch_labels).sum().item()
            
            total_correct += correct
            total_samples += batch_labels.numel()
    
    return total_correct / total_samples


def ask_continue():
    resp = input("Continue training? y/n: ")
    if resp.lower() == 'n':
        return None
    elif resp.lower() == 'y':
        num_epoch = input("Enter number of epochs to train (or 'q' to quit): ")
        if num_epoch.lower() == 'q':
            return None
        try:
            num_epoch = int(num_epoch)
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")
            return ask_continue()
        
        lr = input("Enter learning rate (or 'q' to quit): ")
        if lr.lower() == 'q':
            return None
        try:
            lr = float(lr)
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")
            return ask_continue()
        
        return (num_epoch, lr)
    else:
        print("Invalid input. Please enter 'y' or 'n'.")
        return ask_continue()