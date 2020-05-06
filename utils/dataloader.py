import torch
from torchvision import datasets, transforms
import os


def load_data(dataset_name="car_dataset", input_size=224, batch_size=32, data_dir="./data"):

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    if dataset_name == "car_dataset":
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir + '/car_data', x), data_transforms[x]) for x in
                          ['train', 'val']}
    elif dataset_name == "car_dataset_modified":
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir + '/car_data_modified', x), data_transforms[x]) for x in
                          ['train', 'val']}
    elif dataset_name == "cifar100":
        image_datasets = {"train": datasets.CIFAR100('./data', train=True,
                                                     transform=transforms.Compose(
                                                         [transforms.Resize(224),
                                                          transforms.ToTensor(),
                                                          ]), download=True),
                          "val": datasets.CIFAR100('./data', train=False,
                                                   transform=transforms.Compose(
                                                       [transforms.Resize(224),
                                                        transforms.ToTensor(),
                                                        ]))}
    else:
        image_datasets = {"train": datasets.MNIST('./data', train=True, download=True,
                                                  transform=transforms.Compose(
                                                      [transforms.Resize(224), transforms.Grayscale(3),
                                                       transforms.ToTensor(),
                                                       ])),
                          "val": datasets.MNIST('./data', train=False,
                                                transform=transforms.Compose(
                                                    [transforms.Resize(224), transforms.Grayscale(3),
                                                     transforms.ToTensor(),
                                                     ]))}

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=3) for x in
        ['train', 'val']}

    next(iter(dataloaders_dict['train']))
    return dataloaders_dict
