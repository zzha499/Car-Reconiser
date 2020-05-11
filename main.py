import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utils.model_initializer import *
from utils.data_loader import *
from utils.model_trainer import *
import argparse


parser = argparse.ArgumentParser(description='PyTorch Car Dataset Training')
parser.add_argument('--dataset', '-d', default="car_data_modified", type=str, help='Available Dataset: [car_dataset, '
                                                                                   'car_dataset_modified, cifar100, '
                                                                                   'mnist]')
parser.add_argument('--model', '-m', default="resnet", type=str, help='Available Model: [resnet, vgg, squeezenet, '
                                                                      'densenet, inception]')
parser.add_argument('--lr', '-lr', default=0.05, type=float, help='Starting learning rate')
parser.add_argument('--batch_size', '-b', default=64, type=int, help='Batch size')
parser.add_argument('--num_of_epochs', '-es', default=15, type=int, help='Number of epochs')
parser.add_argument('--save', '-s', default=False, action='store_true', help='Save model')
parser.add_argument('--load', '-l', default=False, action='store_true', help='Load model')
args = parser.parse_args()

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "./data"

# The dataset to train the model on [car_data, car_data_modified, cifar100, mnist]
dataset_name = args.dataset

# Models to choose from [resnet, vgg, squeezenet, densenet, inception]
model_name = args.model

# Number of classes in the dataset
num_classes = len(os.listdir(os.path.join(os.path.join(data_dir, args.dataset), "train")))

# Batch size for training (change depending on how much memory you have)
batch_size = args.batch_size

# Number of epochs to train for
num_epochs = args.num_of_epochs

# Learning Rate of optimizer
learning_rate = args.lr

# gamma of scheduler
gamma = 0.9

# Flag for saving model
save_model = args.save

# Flag for loading model
load_model = args.load


if __name__ == "__main__":
    # Initialize the model for this run
    model, input_size = initialize_model(model_name, num_classes)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Setup scheduler for lr
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

    if load_model:
        if os.path.exists("./saved_models/" + dataset_name + "_" + model_name + ".pt"):
            saved_model = torch.load("./saved_models/" + dataset_name + "_" + model_name + ".pt")
            model, optimizer, learning_rate = saved_model["model"], saved_model["optimizer"], saved_model["lr"]
        else:
            print("No saved model --- new model created")

    # Print the model we just instantiated
    # print(model)

    dataloaders_dict, classes = load_data(dataset_name, input_size, batch_size, data_dir)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.functional.nll_loss()

    # Train and evaluate
    print("Training model:")
    model, thist, vhist = train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs=num_epochs,
                              is_inception=(model_name == "inception"), classes=classes)

    if save_model:
        saved_model = {
            'model': model,
            'optimizer': optimizer,
            'lr': learning_rate
        }
        torch.save(saved_model, "./saved_models/" + dataset_name + "_" + model_name + ".pt")

    # Plot the training curves of validation accuracy vs. number
    #  of training epochs for the the trained model

    thist = [h.cpu().numpy() for h in thist]
    vhist = [h.cpu().numpy() for h in vhist]

    plt.title("Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy")
    plt.plot(range(1, num_epochs + 1), thist, label="Training Accuracy")
    plt.plot(range(1, num_epochs + 1), vhist, label="Validate Accuracy")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.show()
    exit()
