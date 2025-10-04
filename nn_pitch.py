import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch

import data_parser as dp

from nn_models import PitchCNN
from nn_models import ensure_tensor, evaluate, train, ask_continue, WeightedMSELoss, calculate_note_weights, print_note_and_class_metrics
import nn_models

# Create a synthetic dataset where learning is guaranteed
def create_synthetic_data():
    # Simple pattern: first freq bin high = note present
    data = torch.randn(15, 1, 256, 128) * 0.1  # noise
    labels = torch.zeros(15, 88)
    
    # Make it learnable: when first frequency bin > 0.5, note is present
    for i in range(15):
        if torch.rand(1) > 0.5:  # 50% positive examples
            data[i, 0, 0, :] = 1.0  # strong signal in first freq bin
            labels[i, 0] = 1.0  # predict first note
    
    return data, labels

# Create obvious synthetic data
def create_obvious_data():
    data = torch.randn(10, 1, 256, 128) * 0.01  # Mostly noise
    labels = torch.zeros(10, 88)
    
    # Make it OBVIOUS: specific pattern = specific note
    for i in range(10):
        if i < 5:  # First 5 samples: note 0 present
            data[i, 0, 0:10, 0:10] = 10.0  # Very strong signal
            labels[i, 0] = 1.0
        else:  # Last 5 samples: note 1 present  
            data[i, 0, 10:20, 10:20] = 10.0  # Very strong signal
            labels[i, 1] = 1.0
    
    return data, labels

# Test with synthetic data
#synth_data1, synth_labels1 = create_synthetic_data()
#synth_data2, synth_labels2 = create_synthetic_data()
#synth_data3, synth_labels3 = create_synthetic_data()
# If this doesn't learn, the issue is in your training code

def main():
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = dp.load_dataset_from_file(
        'parsed data/shuffled_dataset_seed_0.npz', n_samples=None, shuffle=True)
    #train_data, train_labels, valid_data, valid_labels, test_data, test_labels = synth_data1, synth_labels1, synth_data2, synth_labels2, synth_data3, synth_labels3
    print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
    print(f"Valid data shape: {valid_data.shape}, Valid labels shape: {valid_labels.shape}")
    print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")

    # Convert to tensors
    train_data = ensure_tensor(train_data)
    train_labels = ensure_tensor(train_labels)
    valid_data = ensure_tensor(valid_data)
    valid_labels = ensure_tensor(valid_labels)
    test_data = ensure_tensor(test_data)
    test_labels = ensure_tensor(test_labels)

    print(f"Train data tensor shape: {train_data.shape}, Train labels tensor shape: {train_labels.shape}")

    #train_data = train_data.unsqueeze(1)  # add channel dim
    #valid_data = valid_data.unsqueeze(1)  # add channel dim
    #test_data = test_data.unsqueeze(1)  # add channel dim

    # hyperparameters
    lr = 0.05
    #num_epoch = 0
    lamb = 0#.00005
    n_layers = 5
    threshold = 0.1

    train_accs, val_accs, test_accs = [], [], []

    #torch.manual_seed(0)
    print(f"Torch seed: {torch.seed()}")

    # nn model
    model = PitchCNN(num_notes=88, input_shape=(train_data.shape[1], train_data.shape[2], train_data.shape[3]))

    train_acc = evaluate(model, train_data[:200000], train_labels[:200000])
    val_acc = evaluate(model, valid_data, valid_labels)
    test_acc = evaluate(model, test_data, test_labels)
    print(f"Initial Model \t Train Acc: {train_acc:.4f} \t Valid Acc: {val_acc:.4f} \t Test Acc: {test_acc:.4f}")  

    # See what the model is actually predicting
    initial_predictions = model(train_data[:1000])
    print("Prediction range:", initial_predictions.min().item(), initial_predictions.max().item())
    print("Mean prediction:", initial_predictions.mean().item())
    # If predictions are all near 0, model learned "always predict silence"

    cur_epoch = 0
    delta_loss = 0.0
    while resp := ask_continue():
        num_epoch = resp[0]
        lr = resp[1]
        if num_epoch is None:
            print("Exiting training loop.")
            break
        for epoch in range(num_epoch):
            train_losses, val_accuracies = train(model, lr, lamb, train_data, train_labels, valid_data, valid_labels, num_epoch=1,
                                                  )
            delta_loss -= train_losses[-1]

            train_acc = evaluate(model, train_data[:200000], train_labels[:200000])
            #val_acc = evaluate(model, valid_data, valid_labels)
            val_acc = val_accuracies[-1]
            test_acc = evaluate(model, test_data, test_labels)

            train_accs.append(train_acc)
            val_accs.append(val_acc)
            test_accs.append(test_acc)

            # See what the model is actually predicting
            initial_predictions = model(train_data[:1000])
            print("Prediction range:", initial_predictions.min().item(), initial_predictions.max().item())
            print("Mean prediction:", initial_predictions.mean().item())
            # If predictions are all near 0, model learned "always predict silence"

            print(f"Final Model Epoch {cur_epoch} \t Change in loss: {delta_loss:.12f} \t Train Acc: {train_acc:.4f} \t Valid Acc: {val_acc:.4f} \t Test Acc: {test_acc:.4f}")
            cur_epoch += 1
            delta_loss = train_losses[-1]
        nn_models.visualize_feature_maps(model, train_data[0:1][0])
        print_note_and_class_metrics(valid_labels, model(valid_data))

    torch.save(model.state_dict(), f"cnn_acc{test_acc:.4f}.pth")

    # final model plots
    plt.plot(range(cur_epoch), train_accs, label='Training Accuracy', color='blue')
    plt.plot(range(cur_epoch), val_accs, label='Validation Accuracy', color='green')
    plt.axhline(y=test_acc, linestyle='--', color='red', label='Final Model Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Final Model Accuracies over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{lr} lr, {cur_epoch} epochs, {n_layers} layers.png")


if __name__ == "__main__":
    main()
