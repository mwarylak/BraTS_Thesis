import display_methods as dm
import torch
import os
import data
import json
import models.unet as unet
import models.attentionunet as attention
import models.resnetunet as resnet
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_sample_from_saved_model(model, test_input, test_target, device):
    model.eval()
    test_input, test_target = test_input.to(device), test_target.to(device)

    with torch.no_grad():
        test_pred = torch.sigmoid(model(test_input))

    image = test_input.detach().cpu().numpy().squeeze(0)
    mask_pred = test_pred.detach().cpu().numpy().squeeze(0)
    mask_target = test_target.detach().cpu().numpy().squeeze(0)

    dm.display_mask_channels_as_rgb(mask_pred, title='Predicted Mask Channels as RGB')
    dm.display_mask_channels_as_rgb(mask_target, title='Ground Truth as RGB')
    dm.display_side_by_side(image, mask_pred, mask_target)

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def main():
    
    test_directory = '/content/BraTS_Thesis/Files/Test_Samples'
    with open('/content/BraTS_Thesis/NN_config.json', 'r') as f:
        train_config = json.load(f)

    test_files = [os.path.join(test_directory, f) for f in os.listdir(test_directory) if f.endswith('.h5')]
    test_dataset = data.BrainScanDataset(test_files, normalization=train_config['normalization_type'], deterministic=True)
    
    random_index = random.randint(0, len(test_dataset) - 1)
    test_input, test_target = test_dataset[random_index]
    test_input = test_input.unsqueeze(0)  
    test_target = test_target.unsqueeze(0)  
    
    if train_config['model'] == 'UNet':
        model = unet.Net(train_config['activation_fun']).to(device)  
        model = load_model(model, path='/content/BraTS_Thesis/models_weights/unet_weights.pth')

    elif train_config['model'] == "AttentionUNet":
        model = attention.Net(train_config['activation_fun']).to(device)  
        model = load_model(model, path='/content/BraTS_Thesis/models_weights/attentionunet_weights.pth')

    elif train_config['model'] == "ResNetUNet":
        model = resnet.Net(train_config['activation_fun']).to(device)  
        model = load_model(model, path='/content/BraTS_Thesis/models_weights/resnetunet_weights.pth')

    test_sample_from_saved_model(model, test_input, test_target, device)

if __name__ == "__main__":
    main()