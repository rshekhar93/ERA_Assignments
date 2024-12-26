# CIFAR10 Custom Network Implementation

This project implements a custom CNN architecture for the CIFAR10 dataset using PyTorch with the following specifications:

- Custom architecture with C1C2C3C40 design
- Uses Depthwise Separable Convolution
- Uses Dilated Convolution
- Global Average Pooling (GAP)
- Albumentation augmentations
- Total parameters < 200k
- Target accuracy: 85%

## Requirements
```
bash
pip install -r requirements.txt
```

## Architecture Details
- Input: 32x32x3
- Uses dilated convolutions instead of MaxPooling
- One layer with Depthwise Separable Convolution
- Global Average Pooling
- Final FC layer for classification

## Training
```
bash
python train.py
```

## Results
- Achieved XX% accuracy on test set
- Total parameters: XXX,XXX
- Training time: XX hours