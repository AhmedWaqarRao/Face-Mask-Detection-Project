# Face_mask_Detection
YOLOv3

YOLOv3 is used to complete this project using AlexeyAB Github repo

https://github.com/AlexeyAB/darknet.git

Following dataset from Kaggle is used:

https://www.kaggle.com/andrewmvd/face-mask-detection

LabelImg was used to label the images using YOLO format.

# Changes made in Darknet cfg file
batch=16

subdivisions=4

max_batches=4000

stepsize=3200,3600

classes=2 (In 3 yolo layers)

layers=21 (In 3 convolutional Layers preceding yolo layers)

Darknet Pre-trained weights can be get from the following links

https://pjreddie.com/media/files/darknet53.conv.74

Weights file of the mask_detection model can be accessed with the following link:

https://drive.google.com/file/d/13FZdO7kVMWs_XbJ-T_i5-831CreMObPR/view?usp=sharing

# Results:
2 classes masked and not_masked can be detected with over 90 percent accuracy in various scenarios

![image](https://user-images.githubusercontent.com/58310295/134740812-1416af64-1ba8-4588-82c5-873be5f6991a.png)

