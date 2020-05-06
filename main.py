from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from my_models import resnet, vgg

# print("PyTorch Version: ",torch.__version__)
# print("Torchvision Version: ",torchvision.__version__)

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "./data/car_modified"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 14

# Batch size for training (change depending on how much memory you have)
batch_size = 64

# Number of epochs to train for
num_epochs = 5

# Learning Rate of
learning_rate = 0.1

# The dataset to train the model on [car_dataset, cifar100, mnist]
dataset_name = "car_dataset"

# Flag for saving model
save_model = False

# Flag for loading model
load_model = True

# Flag for testing model
run_test = False

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False


def train_model(model, dataloaders, criterion, optimizer, num_epochs=1, is_inception=False, scheduler=None):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs1, aux_outputs2 = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs1, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if batch_idx % 10 == 0:
                            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch, batch_idx * len(inputs), len(dataloaders["train"].dataset),
                                       100. * batch_idx / len(dataloaders["train"]), loss.item()))

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Accuracy: {:.2f}%'.format(phase, epoch_loss, epoch_acc * 100))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        scheduler.step()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Accuracy: {:2f}%'.format(best_acc * 100))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True


def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    input_size = 0

    if model_name == "googlenet":
        model = models.googlenet(pretrained=use_pretrained)
        model.fc = nn.Linear(1024, num_classes)
        input_size = 224

    elif model_name == "resnet":
        """ Resnet10
        """
        model = resnet.resnet10(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_rs = model.fc.in_features
        model.fc = nn.Linear(num_rs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_rs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_rs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG8_bn
        """
        model = vgg.vgg8(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_rs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_rs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_rs = model.classifier.in_features
        model.classifier = nn.Linear(num_rs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        # Handle the auxilary net
        num_rs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_rs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model, input_size


def test_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # data = torch.squeeze(data)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    # Initialize the model for this run
    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)

    if load_model:
        model = torch.load("./saved_models/" + dataset_name + "_" + model_name + ".pt")[
            "model"]

    set_parameter_requires_grad(model, feature_extract)

    # # Print the model we just instantiated
    # print(model)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
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
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                          ['train', 'val']}

    elif dataset_name == "cifar100":
        image_datasets = {"train": datasets.CIFAR100('./data', train=True,
                                                     transform=transforms.Compose(
                                                         [transforms.Resize(224),
                                                          transforms.ToTensor(),
                                                          ]), download=False),
                          "val": datasets.CIFAR100('./data', train=False,
                                                   transform=transforms.Compose(
                                                       [transforms.Resize(224),
                                                        transforms.ToTensor(),
                                                        ]))}
    else:
        image_datasets = {"train": datasets.MNIST('./data', train=True, download=False,
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

    inputs, classes = next(iter(dataloaders_dict['train']))

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model = model.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model.parameters()
    # print("Params to learn:")
    # if feature_extract:
    #     params_to_update = []
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             params_to_update.append(param)
    #             print("\t", name)
    # else:
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             print("\t", name)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
    # optimizer = optim.Adam(params_to_update, lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    if load_model:
        optimizer = torch.load("./saved_models/" + dataset_name + "_" + model_name + ".pt")[
            "optimizer"]

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.functional.nll_loss()

    if not run_test:
        # Train and evaluate
        print("Training model:")
        model, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs,
                                  is_inception=(model_name == "inception" or model_name == "googlenet"), scheduler=scheduler)

        if save_model:
            saved_model = {
                'model': model,
                'optimizer': optimizer,
                'lr': learning_rate
            }
            torch.save(saved_model,
                       "./saved_models/" + dataset_name + "_" + model_name + ".pt")

        # Plot the training curves of validation accuracy vs. number
        #  of training epochs for the the trained model

        hist = [h.cpu().numpy() for h in hist]

        plt.title("Validation Accuracy vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Validation Accuracy")
        plt.plot(range(1, num_epochs + 1), hist, label=model_name)
        plt.ylim((0, 1.))
        plt.xticks(np.arange(1, num_epochs + 1, 1.0))
        plt.legend()
        plt.show()
    else:
        print("Testing model:")
        test_model(model, device, dataloaders_dict["val"])
