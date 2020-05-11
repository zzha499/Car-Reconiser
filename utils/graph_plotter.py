import torch
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score


def plot_accuracy_vs_epoch(train_acc, val_acc, num_epochs):
    # Plot the training curves of validation accuracy vs. number
    #  of training epochs for the the trained model
    train_acc = [loss.cpu().numpy() for loss in train_acc]
    val_acc = [loss.cpu().numpy() for loss in val_acc]

    plt.figure(1)
    plt.title("Accuracy vs Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy")
    plt.plot(range(1, num_epochs + 1), train_acc, label="Training Accuracy")
    plt.plot(range(1, num_epochs + 1), val_acc, label="Validate Accuracy")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.show()


def plot_loss_vs_epoch(train_loss, val_loss, num_epochs):
    # Plot the training curves of validation accuracy vs. number
    #  of training epochs for the the trained model
    # train_loss = [loss.cpu().numpy() for loss in train_loss]
    # val_loss = [loss.cpu().numpy() for loss in val_loss]

    plt.figure(2)
    plt.title("Loss vs Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1, num_epochs + 1), train_loss, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_loss, label="Validate Loss")
    plt.ylim((0, 6))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.show()


def plot_confusion_matrix(model, dataset, classes, normalize=False):
    with torch.no_grad():
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=64)
        preds = get_all_preds(model, data_loader).to("cpu")
    cm = confusion_matrix(dataset.targets, preds.argmax(dim=1))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(3)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# https://deeplizard.com/learn/video/0LhiS6yu2qQ
@torch.no_grad()
def get_all_preds(model, data_loader):
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds = torch.tensor([]).to(device)
    for batch in data_loader:
        inputs, labels = batch
        inputs = inputs.to(device)
        preds = model(inputs)
        all_preds = torch.cat((all_preds, preds), dim=0)
    return all_preds


def calculate_scores(model, dataset):
    with torch.no_grad():
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=10000)
        preds = get_all_preds(model, data_loader)

    precision, recall, fscore, support = score(dataset.targets, preds)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))



