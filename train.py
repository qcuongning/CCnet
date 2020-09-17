

import DataLoader

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]




def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    num_class = 1
    model = UNet(n_channels=3, n_classes=1).to(device)

    # freeze backbone layers
    #for l in model.base_layers:
    #    for param in l.parameters():
    #        param.requires_grad = False

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=60)
if __name__ == "__main__":
    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
    ])

    train_set = SimDataset(2000, transform=trans)
    val_set = SimDataset(200, transform=trans)

    image_datasets = {
        'train': train_set, 'val': val_set
    }

    batch_size = 25

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }
    main()