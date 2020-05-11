import torch
import torchvision
from torchvision import datasets, transforms
import os
import numpy as np
import matplotlib.pyplot as plt


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

    # check cuda availability
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    # Create training and validation datasets
    if dataset_name == "car_data":
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir + '/car_data', x), data_transforms[x]) for x in
                          ['train', 'val']}
    elif dataset_name == "car_data_modified":
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
    elif dataset_name == "mnist":
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
    else:
        print("UNKNOWN Dataset! Please Chooses from the following datasets:")
        print("\t[car_data, car_data_modified, cifar100, mnist]")
        exit()

    classes = image_datasets["train"].classes
    class_names = {i: name for i, name in enumerate(classes)}
    # print(class_names)

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, **kwargs) for x in
        ['train', 'val']}

    inputs, labels = next(iter(dataloaders_dict['train']))

    # def imshow(inp, title=None, ax=None, figsize=(10, 10)):
    #     """Imshow for Tensor."""
    #     print("ploting images")
    #     inp = inp.numpy().transpose((1, 2, 0))
    #     mean = np.array([0.485, 0.456, 0.406])
    #     std = np.array([0.229, 0.224, 0.225])
    #     inp = std * inp + mean
    #     inp = np.clip(inp, 0, 1)
    #     if ax is None:
    #         fig, ax = plt.subplots(1, figsize=figsize)
    #     ax.imshow(inp)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     if title is not None:
    #         ax.set_title(title)
    #
    # # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs, nrow=4)
    #
    # fig, ax = plt.subplots(1, figsize=(10, 10))
    # title = [class_names[x.item()] if (i + 1) % 4 != 0 else class_names[x.item()] + '\n' for i, x in enumerate(labels)]
    # imshow(out, title=' | '.join(title), ax=ax)

    return dataloaders_dict, class_names, image_datasets
