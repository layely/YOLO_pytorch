# YOLOv1

Reimplementation of YOLOv1 with a resnet50 backbone in Pytorch (wip).

## Requirment
albumentations  0.4.4                        
imgaug          0.2.6                            
numpy           1.17.3                        
opencv-python   4.1.2.30                                        
scikit-image    0.16.2             
scipy           1.4.1                           
tensorboard     2.0.0              
pytorch           1.3.1              
torchvision     0.4.2                         
tqdm            4.40.0             

## Train
* Open train.py
* go to main function and adapt to your need and dataset emplacement below lines:
```
 # Train and test files
voc_2007 = "/home/layely/Myprojects/datasets/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages"
voc_2012 = "/home/layely/Myprojects/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"
train_images_dir = [voc_2007, voc_2012]
train_files = ["voc_2007.txt", "voc_2012.txt"]
voc_2007_test = "/home/layely/Myprojects/datasets/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages"
test_images_dir = [voc_2007_test]
test_files = ["voc_2007_test.txt"]

channels, height, width = (3, 448, 448)
S = 7  # SxS grid cells
B = 2  # Number of bounding boxes per cell
C = 20  # Number of classes

# Training hyperparameters
epochs = 160
lr = 0.01
momentum = 0.9
weight_decay = 5e-4
opt = torch.optim.SGD
batch_size = 24 * num_gpu


# Image normalization parameters
# Note that images are squished to
# the range [0, 1] before normalization
mean = [0.485, 0.456, 0.406] # RGB - Imagenet means
std = [0.229, 0.224, 0.225] # RGB - Imagenet standard deviations

...
* run train.py
```

# Test results
* open test.py and adapt:
```
voc_2007_test = "/home/layely/Myprojects/datasets/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages"
test_images_dir = [voc_2007_test]
test_files = ["voc_2007_test.txt"]
```
* run test.py
* result will be saved in ./test_results
