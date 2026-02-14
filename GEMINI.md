# Project Overview

This project is a desktop application for car brand recognition. It's built with Python and uses the PyQt5 library for the graphical user interface. The application allows users to load an image, and it will then detect and identify the brand of any cars present in the image.

The core of the application is a deep learning pipeline that consists of two main components:

1.  **Object Detection:** A pre-trained RetinaNet model (ResNet50 backbone) is used to detect the location of cars in the input image.
2.  **Image Classification:** An ensemble of two models is used to classify the brand of each detected car:
    *   A custom-built Convolutional Neural Network (CNN).
    *   A transfer learning model based on EfficientNet.

The application displays the original image alongside the processed image with bounding boxes and labels indicating the predicted car brand and confidence score.

## Building and Running

### Dependencies

The project requires Python and several libraries. You can install the dependencies using pip:

```bash
pip install torch torchvision opencv-python Pillow PyQt5
```

### Running the Application

To run the application, execute the `main.py` script:

```bash
python main.py
```

This will launch the GUI. From there, you can load an image and run the recognition process.

### Training the Models

The classification models can be trained using the `train.py` script. This script expects a dataset to be present in a `dataset` directory, with subdirectories for `train`, `val`, and `test` sets.

To start the training process, run:

```bash
python train.py
```

The script will prompt you to choose which model to train (the custom CNN, the EfficientNet transfer learning model, or both). The trained models will be saved in the `models/` directory.

## Development Conventions

*   **Models:** The trained models are stored in the `models/` directory. The application expects to find `model_ft.pt` (the custom CNN) and `model_tl.pt` (the EfficientNet model) in this directory to function correctly.
*   **Dataset:** The training script expects the dataset to be organized in an ImageFolder structure within the `dataset/` directory.
*   **Configuration:** The `utils/class_id.json` file is generated during training and maps the class indices used by the models to human-readable class names (the car brands).
*   **Core Logic:** The core deep learning logic for detection and classification is encapsulated in the `utils/core.py` file.
*   **GUI:** The main application window and GUI logic are defined in `main.py`.
