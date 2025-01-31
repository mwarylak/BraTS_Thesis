import torch
import numpy as np
import matplotlib.pyplot as plt
import display_methods_NN as dm
from torch import nn
import os
import data
from torch.utils.data import DataLoader
import models.unet as unet


def dice_coefficient(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)

    preds = (preds > 0.5).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

    return dice.item()

def calculate_precision(preds, targets, smooth=1e-6):

    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    preds = preds.view(-1)
    targets = targets.view(-1)

    true_positives = (preds * targets).sum()
    predicted_positives = preds.sum()

    precision_value = (true_positives + smooth) / (predicted_positives + smooth)
    return precision_value.item()

def calculate_recall(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    preds = preds.view(-1)
    targets = targets.view(-1)

    true_positives = (preds * targets).sum()
    actual_positives = targets.sum()

    recall_value = (true_positives + smooth) / (actual_positives + smooth)
    return recall_value.item()

def calculate_accuracy(preds, targets):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    preds = preds.view(-1)
    targets = targets.view(-1)

    correct = (preds == targets).float().sum()
    total = targets.numel()

    accuracy_value = correct / total
    return accuracy_value.item()

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total Parameters: {total_params:,}\n')

def save_model(model, path='model_weights.pth'):
    torch.save(model.state_dict(), path)

class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=1., weight_bce=0.):
        super(CombinedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.bce_loss = nn.BCEWithLogitsLoss()  # Wagi dla każdej klasy

    def forward(self, pred, target):
        # Entropia krzyżowa
        bce_loss_value = self.bce_loss(pred, target)
        # Dice Loss
        dice_loss_value = 1 - dice_coefficient(pred, target)
        # Połączona strata
        return self.weight_bce * bce_loss_value + self.weight_dice * dice_loss_value

def train_model(model, train_dataloader, val_dataloader, config, verbose=True, loss_function='BCE', model_saving='True'):
    device = config['device']
    n_epochs = config['n_epochs']
    learning_rate = config['learning_rate']
    batches_per_epoch = config['batches_per_epoch']
    lr_decay_factor = config['lr_decay_factor']

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if loss_function == 'Combined':
      print('Combined')
      loss_fn = CombinedLoss(weight_dice=0.5, weight_bce=0.5)

    elif loss_function == 'Dice':
      print('Dice Loss')
      loss_fn = CombinedLoss(weight_dice=1., weight_bce=0.)

    else:
      print('Binary Cross Entropy Loss')
      loss_fn = nn.BCEWithLogitsLoss()

    train_epoch_losses = []
    val_epoch_losses = []

    print("Training...")
    for epoch in range(1, n_epochs + 1):
        # Decay learning rate
        current_lr = learning_rate * (lr_decay_factor ** (epoch - 1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Training step
        model.train()
        train_epoch_loss = 0
        total_dice, total_precision, total_recall, total_accuracy = 0, 0, 0, 0
        for train_batch_idx, (train_inputs, train_targets) in enumerate(train_dataloader, start=1):
            if verbose:
                print(f"\rTrain batch: {train_batch_idx}/{batches_per_epoch}, Avg batch loss: {train_epoch_loss/train_batch_idx:.6f}", end='')

            train_inputs = train_inputs.to(device)
            train_targets = train_targets.to(device)

            # Forward pass
            train_preds = model(train_inputs)
            train_batch_loss = loss_fn(train_preds, train_targets)
            train_epoch_loss += train_batch_loss.item()

            optimizer.zero_grad()
            train_batch_loss.backward()
            optimizer.step()

            # Calculate metrics
            dice_score = dice_coefficient(train_preds, train_targets)
            precision = calculate_precision(train_preds, train_targets)
            recall = calculate_recall(train_preds, train_targets)
            accuracy = calculate_accuracy(train_preds, train_targets)

            total_dice += dice_score
            total_precision += precision
            total_recall += recall
            total_accuracy += accuracy

            if train_batch_idx >= batches_per_epoch:
                if verbose: print()
                break

        # Average metrics for the epoch
        avg_train_loss = train_epoch_loss / batches_per_epoch
        avg_dice = total_dice / batches_per_epoch
        avg_precision = total_precision / batches_per_epoch
        avg_recall = total_recall / batches_per_epoch
        avg_accuracy = total_accuracy / batches_per_epoch

        print(f"\nEpoch {epoch}/{n_epochs}:")
        print(f"Training Loss: {avg_train_loss:.4f}, Dice: {avg_dice:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, Accuracy: {avg_accuracy:.4f}")

        train_epoch_losses.append(avg_train_loss)

        # Validation step
        avg_val_loss, avg_val_dice, avg_val_precision, avg_val_recall, avg_val_accuracy = validate_model(model, val_dataloader, loss_fn, device)
        print(f"Validation Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}, Precision: {avg_val_precision:.4f}, Recall: {avg_val_recall:.4f}, Accuracy: {avg_val_accuracy:.4f}")
        val_epoch_losses.append(avg_val_loss)

        if epoch % 5 == 0 and model_saving:
            path = f'/content/drive/MyDrive/Praca_Magisterska/Modele/attention_unet_weights_{epoch}.pth'
            save_model(model, path)
            print(f"Model saved at epoch {epoch} to {path}")

    return train_epoch_losses, val_epoch_losses

def validate_model(model, val_dataloader, loss_fn, device):

    model.eval()
    val_loss = 0.0
    dice_total = 0.0
    precision_total = 0.0
    recall_total = 0.0
    accuracy_total = 0.0
    n_batches = len(val_dataloader)

    with torch.no_grad():  
        for val_inputs, val_targets in val_dataloader:
            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device)

            # Forward pass
            val_preds = model(val_inputs)

            # Loss calculation
            loss = loss_fn(val_preds, val_targets)
            val_loss += loss.item()

            # Metric calculations
            dice_total += dice_coefficient(val_preds, val_targets)
            precision_total += calculate_precision(val_preds, val_targets)
            recall_total += calculate_recall(val_preds, val_targets)
            accuracy_total += calculate_accuracy(val_preds, val_targets)

    val_loss /= n_batches
    dice_total /= n_batches
    precision_total /= n_batches
    recall_total /= n_batches
    accuracy_total /= n_batches

    return val_loss, dice_total, precision_total, recall_total, accuracy_total


def plot_learning_curves(train_epoch_losses, val_epoch_losses):
    fig, axis = plt.subplots(1, 1, figsize=(10, 6))

    # Plot training and validation loss (NaN is used to offset epochs by 1)
    axis.plot([np.NaN] + train_epoch_losses, color='#636EFA', marker='o', linestyle='-', linewidth=2, markersize=5, label='Training Loss')
    axis.plot([np.NaN] + val_epoch_losses,   color='#EFA363', marker='s', linestyle='-', linewidth=2, markersize=5, label='Validation Loss')

    # Adding title, labels and formatting
    axis.set_title('Training and Validation Loss Over Epochs', fontsize=16)
    axis.set_xlabel('Epoch', fontsize=14)
    axis.set_ylabel('Loss', fontsize=14)

    axis.set_ylim(0, 0.02)

    axis.legend(fontsize=12)
    axis.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def display_test_sample(model, test_input, test_target, device):
    test_input, test_target = test_input.to(device), test_target.to(device)

    test_pred = torch.sigmoid(model(test_input))

    image = test_input.detach().cpu().numpy().squeeze(0)
    mask_pred = test_pred.detach().cpu().numpy().squeeze(0)
    mask_target = test_target.detach().cpu().numpy().squeeze(0)

    dm.display_mask_channels_as_rgb(mask_pred, title='Predicted Mask Channels as RGB')
    dm.display_mask_channels_as_rgb(mask_target, title='Ground Truth as RGB')
    dm.display_side_by_side(image, mask_pred, mask_target)


def main():
    
    train_config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_epochs': 50,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'batches_per_epoch': 64,
    'lr_decay_factor': 1,
    'normalization_type': '', #
    'augmentation_types': [],
    'model_saving': True, 
    'loss_function': 'BCE',
    'activation_fun': 'ReLU',
    'model': 'UNet' 
    }

    directory = "/content/BraTS2020_training_data/content/data"
    test_directory = '/BraTS_Thesis/Files/Test_Samples'

    h5_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
    print(f"Found {len(h5_files)} .h5 files:\nExample file names:{h5_files[:3]}")

    plt.style.use('ggplot')
    plt.rcParams['figure.facecolor'] = '#FFFFFF'
    plt.rcParams['text.color']       = '#545353'

    augmentation_types = train_config['augmentation_types'] #augmentation type: NEC_ET, random, mixup lub cutmix
    normalization = train_config['normalization_type'] #normalization type: min-max, z-score lub percent

    with open("/BraTS_Thesis/Files/h5files.txt", "r") as file:
        lesion_files = [line.strip() for line in file]

    #Split data for train, valid and tests
    training_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.h5')] #h5_files_copy]
    np.random.seed(42)
    np.random.shuffle(training_files)

    test_files = [os.path.join(test_directory, f) for f in os.listdir(test_directory) if f.endswith('.h5')]
    test_file_names = [os.path.basename(file) for file in test_files]

    split_idx = int(0.1 * len(training_files))
    val_files = training_files[:split_idx]
    train_files = training_files[split_idx:]
    train_files = [file for file in train_files if os.path.basename(file) not in test_file_names]
    val_files = [file for file in val_files if os.path.basename(file) not in test_file_names]

    #Perform augmentations
    if augmentation_types != []:
        for augmentation_type in augmentation_types:
            for image in train_files[:10]:
                filename = os.path.basename(image)
                if filename in lesion_files:
                    if augmentation_type == 'NEC_ET':
                        augmented_image = data.NEC_ET_augmentation(filename, directory, directory)
                    if augmentation_type == 'random':
                        augmented_image = data.random_augmentation(filename, directory, directory)
                    if augmentation_type == 'mixup':
                        augmented_image = data.rocess_mixup(filename, directory, directory, train_files)
                    if augmentation_type == 'cutmix':
                        augmented_image = data.process_cutmix(filename, directory, directory, train_files)

                    if augmented_image != '':
                        train_files.append(augmented_image)


    # Create the datasets
    train_dataset = data.BrainScanDataset(train_files, normalization=normalization)
    val_dataset = data.BrainScanDataset(val_files, normalization = normalization, deterministic=True)
    test_dataset = data.BrainScanDataset(test_files, normalization=normalization, deterministic=True)

    # Sample dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=5, shuffle=False)

    # Use this to generate test images to view later
    test_input_iterator = iter(DataLoader(test_dataset, batch_size=1, shuffle=False))

    torch.cuda.empty_cache()
    model = unet.UNet(train_config['activation_fun'])
    count_parameters(model)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)

    # Train model
    train_epoch_losses, val_epoch_losses = train_model(model, train_dataloader, val_dataloader, train_config, verbose=True, loss_function=train_config['loss_function'], model_saving=train_config['model_saving'])

    plot_learning_curves(train_epoch_losses, val_epoch_losses)
    # Set so model and these images are on the same device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get an image from the validation dataset that the model hasn't been trained on
    test_input, test_target = next(test_input_iterator)
    display_test_sample(model, test_input, test_target, device)

if __name__ == "__main__":
    main()
