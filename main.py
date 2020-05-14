import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import model_initializer, data_loader, model_trainer, graph_plotter
import argparse

parser = argparse.ArgumentParser(description='PyTorch Car Dataset Training')
parser.add_argument('--dataset', '-d', default="car_data_modified", type=str, help='Available Dataset: [car_dataset, '
                                                                                   'car_dataset_modified, cifar100, '
                                                                                   'mnist]')
parser.add_argument('--model', '-m', default="inception", type=str, help='Available Model: [alexnet, resnet, vgg, '
                                                                      'squeezenet, densenet, inception]')
parser.add_argument('--lr', '-lr', default=0.01, type=float, help='Starting learning rate')
parser.add_argument('--batch_size', '-b', default=16, type=int, help='Batch size')
parser.add_argument('--num_of_epochs', '-es', default=10, type=int, help='Number of epochs')
parser.add_argument('--save_model', '-sm', default=True, action='store_true', help='Save model')
parser.add_argument('--load_model', '-lm', default=True, action='store_true', help='Load model')
# parser.add_argument('--confusion_matrix', '-cm', default=True, action='store_true', help='Plot confusion Matrix')
# parser.add_argument('--accuracy_vs_epoch', '-ae', default=True, action='store_true', help='Plot confusion Matrix')
# parser.add_argument('--loss_vs_epoch', '-le', default=True, action='store_true', help='Plot confusion Matrix')
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
save_model = args.save_model

# Flag for loading model
load_model = args.load_model

# Flag for training model
train_model = True

if __name__ == "__main__":
    # Initialize the model for this run
    model, input_size = model_initializer.initialize_model(model_name, num_classes)

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

    dataloaders_dict, classes, image_datasets, class_names = data_loader.load_data(dataset_name, input_size, batch_size, data_dir)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.functional.nll_loss()

    if train_model:
        # Train and evaluate
        print("Training model:")
        model, train_acc, val_acc, train_loss, val_loss = model_trainer.train_model(model, dataloaders_dict, criterion,
                                                                                    optimizer, scheduler,
                                                                                    num_epochs=num_epochs,
                                                                                    is_inception=(
                                                                                                model_name == "inception"),
                                                                                    classes=classes)

        if save_model:
            saved_model = {
                'model': model,
                'optimizer': optimizer,
                'lr': learning_rate
            }
            torch.save(saved_model, "./saved_models/" + dataset_name + "_" + model_name + ".pt")

        # Plot training and validation losses vs epochs
        graph_plotter.plot_loss_vs_epoch(train_loss, val_loss, num_epochs)

        # Plot training and validation accuracies vs epochs
        graph_plotter.plot_accuracy_vs_epoch(train_acc, val_acc, num_epochs)

    # Plot the confusion matrix and calculate the precision, recall, and F1 scores of the trained model
    graph_plotter.plot_confusion_matrix(model, image_datasets['val'], class_names, normalize=False, Score=True)

    # Exit program
    exit()
