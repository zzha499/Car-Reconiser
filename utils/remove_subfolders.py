import os
import shutil

# The following code was used to modify the structure of the modified dataset to fit the TorchVision ImageFolder structure
dir = 'C:/Users/ziang/Desktop/COMPSYS_302_Project_1/data/car_data_modified/train'

folders = [f.path for f in os.scandir(dir) if f.is_dir()]
for folder in folders:

    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    for sub in subfolders:
        for f in os.listdir(sub):
            src = os.path.join(sub, f)
            dst = os.path.join(folder, f)
            shutil.move(src, dst)
            print("moved --- " + src + " to " + dst)
        shutil.rmtree(sub)


