import torch.nn as nn
from torchvision import models
from .my_models import resnet, vgg


def initialize_model(model_name, num_classes):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    input_size = 0

    if model_name == "resnet":
        """ Resnet10
        """
        model = resnet.resnet10()
        num_rs = model.fc.in_features
        model.fc = nn.Linear(num_rs, num_classes)
        input_size = 224

    elif model_name == "googlenet":
        model = models.googlenet()
        model.fc = nn.Linear(1024, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model = models.alexnet()
        num_rs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_rs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG8_bn
        """
        model = vgg.vgg8()
        num_rs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_rs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model = models.squeezenet1_0()
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model = models.densenet121()
        num_rs = model.classifier.in_features
        model.classifier = nn.Linear(num_rs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model = models.inception_v3()
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