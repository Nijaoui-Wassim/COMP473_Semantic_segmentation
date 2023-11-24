
# Zero-Shot Semantic Segmentation on ADE20K using TensorFlow and GluonCV

## Introduction
This project implements a zero-shot semantic segmentation model on the ADE20K dataset using TensorFlow and GluonCV. It aims to recognize and segment objects in images, even those categories not seen during training.
The aim is to implement and improve upon the **Uncertainty-Aware Learning for Zero-Shot Semantic Segmentation** paper

## Features
- **Model Architecture**: DeepLabV3 for images and Word2Vec for text.
- **Zero-Shot Learning**: Explanation of how your model achieves zero-shot learning.
- **Datasets**: Use of the ADE20K dataset
- **Frameworks**: Utilization of GluonCV / MXNet.

## Getting Started
### Prerequisites
- Python 3.6.13
- gluoncv 0.11.0
- mxnet >= 1.6.0

### Installation
Provide step-by-step instructions to set up the environment and install the necessary dependencies.

```bash
# Clone this repo

# Install dependencies
pip install gluoncv[full] mxnet>=1.6.0 --upgrade

# run the ipynb notebook
```

### Dataset Preparation
uncomment the first lines of preprocessing in the notebook for the first run the comment them again

### Usage and Training the Model
Training is provided as well in the notebook

## License
State the license under MIT license

## Acknowledgments
- Credits to any third-party assets or code used such as Gluon-CV.
- Acknowledgments to any collaborators or contributors including our team members from COMP 473.

## Contact
nijaouiwassim1@yahoo.com
