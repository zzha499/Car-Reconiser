# COMPSYS_302_Project_1 - Car Recogniser
By Will Zhang and Eric Zhao

This repository contains the data loader, models, trainer, and other useful tools for the car recogniser.     
The car recogniser is used to identify the brand of a car from a car image.

# Dependencies
This recogniser require Python 3.6 and Pytorch to be installed.     
Training using GPU is preferred for maximum training speed.

## Dataset - Stanford Cars Dataset
Original Dataset Download Link: https://ai.stanford.edu/~jkrause/cars/car_dataset.html  
Modified Dataset Download Link: https://drive.google.com/file/d/11bS7Az-x4WkMUM066KgAhyiVWx-BqGwa/view?usp=sharing 

Structure of dataset folder for training (TorchVision ImageFolder structure):
<pre>
data/
    car_dataset/
        train/
            Audi/
                001.jpeg
                002.jpeg
                ...
            Ferrari/
            ...
        val/
    car_data_modified/
        train/
        val/
    ...
</pre>   

## Models 
Resnet-10   
Resnet-18   
Alexnet     
Inception

## How To Use

Clone the repository  
Make sure dependencies are installed  
Download required dataset  
Train a specified model using the dataset  
Start recognising cars!  

## Project Report 
Link: https://docs.google.com/document/d/1l-OuXX4TLHf-OyYQ3MRouXOnflG4h4wASf4gs9FDzZ0/edit?usp=sharing
