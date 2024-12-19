# CNN Models on MNIST Data

This repository demonstrates a systematic approach to building and optimizing Convolutional Neural Networks (CNNs) through hands-on implementation. Using the MNIST dataset as our foundation, we progressively develop three CNN models, each introducing more sophisticated techniques and architectural improvements. The project serves as a practical exploration of CNN development, from establishing basic model structure to implementing advanced techniques like data augmentation, batch normalization, and learning rate scheduling.

Starting with a basic CNN implementation to ensure correct model skeleton, we then advance to explore regularization techniques and efficient parameter usage. The final model showcases advanced concepts including skip connections, strategic data augmentation, and dynamic learning rate adjustment. Each iteration not only improves model performance but also demonstrates key concepts in deep learning such as avoiding overfitting, improving model generalization, and achieving consistent high accuracy with minimal parameters. This hands-on approach provides valuable insights into CNN architecture design, model debugging, and performance optimization techniques.

## Notebooks

### 1. Basic Model
- **Notebook**: [MNIST_CNN_Model_1.ipynb](notebooks/MNIST_CNN_Model_1.ipynb)
- **Features**:
  - Basic CNN architecture with 8 convolutional layers
  - Simple MaxPooling and ReLU activations
  - No normalization or regularization techniques
  - Focus on establishing correct model skeleton
- **Architecture Highlights**:
  - Input → Conv layers → MaxPool → Conv layers → Output
  - Uses large kernel (7x7) for final classification

### 2. Advanced Model with Regularization
- **Notebook**: [MNIST_CNN_Model_2.ipynb](notebooks/MNIST_CNN_Model_2.ipynb)
- **Features**:
  - Added Batch Normalization after convolutions
  - Introduced Dropout (0.1) for regularization
  - Replaced large kernel with Global Average Pooling
  - Reduced total parameters while maintaining performance
- **Architecture Improvements**:
  - Better feature extraction with normalized activations
  - More efficient parameter usage
  - Improved model generalization

### 3. Final Model with Advanced Techniques
- **Notebook**: [MNIST_CNN_Fine_Tunning_Model_3.ipynb](notebooks/MNIST_CNN_Fine_Tunning_Model_3.ipynb)
- **Features**:
  - Advanced data augmentation:
    - Random Rotation (-7° to 7°)
    - Random Affine transforms
    - Random Perspective changes
  - Optimized architecture with skip connections
  - Learning rate scheduling with ReduceLROnPlateau
  - Consistently achieves >99.4% test accuracy
- **Training Details**:
  - 15 epochs
  - Batch size: 64
  - Initial learning rate: 0.03
  - Momentum: 0.95

## Results

### Basic Model (Model 1)
- **Parameters**: 8,856
- **Best Training Accuracy**: 98.82%
- **Best Test Accuracy**: 98.67%
- **Analysis**: Good baseline performance without advanced techniques

### Advanced Model (Model 2)
- **Parameters**: 5,672
- **Best Training Accuracy**: 98.74%
- **Best Test Accuracy**: 99.17%
- **Analysis**: Improved generalization with fewer parameters

### Final Model (Model 3)
- **Parameters**: 7,480
- **Best Training Accuracy**: 98.84%
- **Best Test Accuracy**: 99.49%
- **Key Achievements**:
  - Consistently maintains >99.4% test accuracy from epoch 5 onwards
  - Meets all target criteria:
    - Test accuracy > 99.4%
    - Training completed in ≤15 epochs
    - Parameters < 8000
  - Most stable and robust performance among all models

## Requirements
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm