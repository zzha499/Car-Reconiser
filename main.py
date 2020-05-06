from __future__ import print_function
from __future__ import division
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from .model_initializer import *
from .dataloader import *
from .model_trainer import *


# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "./data"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 40

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for
num_epochs = 4

# Learning Rate of optimizer
learning_rate = 0.01

# gamma of scheduler
gamma = 0.7

# The dataset to train the model on [car_dataset, car_dataset_modified, cifar100, mnist]
dataset_name = "car_dataset_modified"

# Flag for saving model
save_model = True

# Flag for loading model
load_model = False

# Flag for testing model
run_test = False


if __name__ == "__main__":
    # Initialize the model for this run
    model, input_size = initialize_model(model_name, num_classes)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = optim.Adam(params_to_update, lr=learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

    if load_model:
        saved_model = torch.load("./saved_models/" + dataset_name + "_" + model_name + ".pt")["model"]
        model, optimizer, learning_rate = saved_model["model"], saved_model["optimizer"], saved_model["lr"]

    # Print the model we just instantiated
    print(model)

    dataloaders_dict = load_data(dataset_name, input_size, batch_size, data_dir)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.functional.nll_loss()

    # Train and evaluate
    print("Training model:")
    model, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs,
                              is_inception=(model_name == "inception"))

    if save_model:
        saved_model = {
            'model': model,
            'optimizer': optimizer,
            'lr': learning_rate
        }
        torch.save(saved_model, "./saved_models/" + dataset_name + "_" + model_name + ".pt")

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
