# Face_mask_Detection
YOLOv3

YOLOv3 is used to complete this project using the AlexeyAB Github repo

https://github.com/AlexeyAB/darknet.git

The following dataset from Kaggle is used:

https://www.kaggle.com/andrewmvd/face-mask-detection

LabelImg was used to label the images using YOLO format.

# Changes made in Darknet cfg file
batch=16

subdivisions=4

max_batches=4000

stepsize=3200,3600

classes=2 (In 3 yolo layers)

layers=21 (In 3 convolutional Layers preceding yolo layers)

Darknet Pre-trained weights can be obtained from the following links

https://pjreddie.com/media/files/darknet53.conv.74

The weights file of the mask_detection model can be accessed with the following link:

https://drive.google.com/file/d/13FZdO7kVMWs_XbJ-T_i5-831CreMObPR/view?usp=sharing

# Results:
2 classes masked and not_masked can be detected with over 90 percent accuracy in various scenarios

![image](https://user-images.githubusercontent.com/58310295/134740812-1416af64-1ba8-4588-82c5-873be5f6991a.png)


**Face Mask Detection Using YOLOv3**

In this project, I implemented a face mask detection system using the YOLOv3 (You Only Look Once, Version 3) model. The goal of this project was to accurately detect whether individuals in images are wearing face masks or not, which is crucial for ensuring public safety, especially during health crises like the COVID-19 pandemic. This project utilized the YOLOv3 model from the AlexeyAB GitHub repository and employed a dataset from Kaggle for training and testing.

**Dataset and Preprocessing**

The dataset used for this project was the Face Mask Detection dataset from Kaggle, which contains labeled images of people with and without face masks. The dataset was chosen for its comprehensive coverage and quality, which are essential for training an accurate model. The images were labeled using LabelImg, a graphical image annotation tool, to create bounding boxes around faces and categorize them into two classes: masked and not_masked. This labeling followed the YOLO format, which specifies the object class and the coordinates of the bounding box.

**Model Architecture**

The YOLOv3 model is well-suited for real-time object detection tasks due to its balance between speed and accuracy. The model's architecture includes multiple convolutional layers followed by YOLO layers that predict bounding boxes and class probabilities. For this project, several modifications were made to the YOLO configuration file (yolov3.cfg) to tailor it to the face mask detection task:

- `batch=16`: This parameter sets the number of images processed in a single batch during training.
- `subdivisions=4`: This parameter divides the batch into smaller mini-batches to fit into GPU memory.
- `max_batches=4000`: This sets the maximum number of training iterations.
- `stepsize=3200,3600`: These are the points at which the learning rate is reduced.
- `classes=2`: This specifies that the model will detect two classes (masked and not_masked).
- `filters=21`: This is set in the convolutional layers preceding the YOLO layers, calculated as (classes + 5) * 3.

**Training Process**

The training process involved using pre-trained weights (darknet53.conv.74) to initialize the model. This approach leverages the knowledge learned from a larger dataset, speeding up the training process and improving accuracy. The training was conducted using the Darknet framework, a high-performance open-source neural network framework written in C and CUDA. The command to start the training was executed in the terminal as follows:

```bash
./darknet detector train data/obj.data cfg/yolov3-voc_mine.cfg darknet53.conv.74
```

**Results**

The face mask detection model achieved impressive results, with an accuracy of over 90% in various scenarios. The model was able to accurately detect both masked and not_masked faces, demonstrating its effectiveness. The performance was evaluated using standard metrics such as precision, recall, and the F1 score, providing a comprehensive assessment of the model's capabilities.

**Conclusion**

The Face Mask Detection project using YOLOv3 showcases the effectiveness of using advanced deep learning models for real-time object detection tasks. By utilizing the YOLOv3 model and a well-labeled dataset, the project was able to develop a highly accurate face mask detection system. This system can be deployed in various settings to ensure compliance with mask-wearing policies and enhance public safety. The project highlights the importance of proper data annotation, model configuration, and training processes in developing robust machine learning applications.
