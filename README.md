# Catching Z’s: A Deep Dive into Drowsiness Detection

## Overview

AwakeDrowsy is a project aimed at enhancing road safety by employing computer vision techniques to detect driver drowsiness in real-time. The project employs two distinct approaches:

### 1. Real-Time Object Detection (YOLOv5)

Utilizing the YOLOv5 (You Only Look Once) model, AwakeDrowsy performs real-time object detection on a live video feed, focusing on facial recognition. The model is trained to identify and track facial features, enabling the system to monitor the driver's face continuously. This approach provides a robust foundation for understanding driver behavior and detecting signs of drowsiness.

### 2. Image Classification (ResNet18)

AwakeDrowsy also incorporates an image classification model based on ResNet18 architecture. This model is trained on a dataset containing images of drivers categorized as awake or drowsy. The system takes advantage of InsightFace to crop faces from images, and the ResNet18 model classifies these cropped faces in real-time. The classification results are then used to assess the driver's alertness.

## Key Features

- **Real-Time Object Detection:** YOLOv5 model identifies and tracks facial features in live video feed.
- **Image Classification:** ResNet18 model classifies cropped faces into awake or drowsy categories.
- **Safety Enhancement:** Continuous monitoring for signs of drowsiness helps prevent potential accidents.

## Tech Stack

- **Object Detection:** YOLOv5
- **Image Classification:** ResNet18
- **Facial Recognition:** InsightFace
- **Deep Learning Framework:** PyTorch
- **Computer Vision Library:** OpenCV
- **Live Feed Processing:** Python

## Usage

### Real-Time Object Detection Implementation

For real-time object detection (YOLOv5) please see/run `RealTimeObjDetec.ipynb` and for image classification model (ResNet18) please see/run `ImgClassif.ipynb`.

To run best YOLOv5 model, run `import statement` and then load in model 6 by running the block containing:

    model_6 = torch.hub.load('ultralytics/yolov5', 'custom', path='.../AwakeDrowsy/yolov5/runs/train/exp13/weights/last.pt')`

and then run the final block in the file to open live feed detection.

### Image Classification Implementation

To run best image classification model (ResNet18), run import statement and run the following blocks containing:

    data = '.../AwakeDrowsy/data/Images'

    output_folder = '.../AwakeDrowsy/data/Images_cropped'

then,

    train_data = '/Users/oscaramirmansour/AwakeDrowsy/data/Images_cropped/train'

    test_data = '/Users/oscaramirmansour/AwakeDrowsy/data/Images_cropped/test'

    valid_data = '/Users/oscaramirmansour/AwakeDrowsy/data/Images_cropped/validation'

followed by,

    transform_pipeline_train = transforms.Compose([

      transforms.Resize((320, 320)), # want to ensure consistent image sizes
      transforms.RandomRotation(degrees=30),
      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
      transforms.RandomPerspective(distortion_scale=0.5, p=0.6),
      transforms.ToTensor(),
      #transforms.Normalize(mean=mean, std=std)

    ])

    transform_pipeline_train = transforms.Compose([`

      transforms.Resize((320, 320)), # want to ensure consistent image sizes
      transforms.RandomRotation(degrees=30),
      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
      transforms.RandomPerspective(distortion_scale=0.5, p=0.6),
      transforms.ToTensor(),
      #transforms.Normalize(mean=mean, std=std)

    ])

    transform_pipeline_valid = transforms.Compose([

      transforms.Resize((320, 320)),
      transforms.ToTensor(),
      #transforms.Normalize(mean=mean, std=std)

    ])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) 
    # unlike with training data, we dont want to shuffle test data so as to not introduce any potential variability in results and also because real world data isnt shuffled and arrives in order.

    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

And then to run the best performing model (model 3), under the 'Loading & Evaluating Model 3' header, you'll need to specify the model's architecture by runing the code block:

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) 
    # unlike with training data, we dont want to shuffle test data so as to not introduce any potential variability in results and also because real world data isnt shuffled and arrives in order.

    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

Load in the model state from the resnet saved results, by running the following line:

    model_3.load_state_dict(torch.load('resnet_results/AwakeDrowsyResnet18_4.pth'))

Followed by the block which enables cuda (if you dont have a GPU you'll be utilising you CPU):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_3.to(device)

And then finally, run the last 3 code blocks in the file, starting with:

    live_detection_transform = transform_pipeline_test

## Future Developments

While both models are usable and working, there is still much to explore and develop - To begin, more images (diverse range) will be added. Both implementations could do with further parameter tweaking especially the image classification model where architecture and hyper-parameter tweaking proved to be quite time-consuming and requires continual iteration in order to hone-in.

NOTE: Both models were trained locally and without a GPU - more performant hardware is expected and thus faster iteration/development. Stay tuned.



