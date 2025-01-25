# CNN for Image Classification

This project implements a Convolutional Neural Network (CNN) for image classification tasks, achieving an accuracy of 77% on the CIFAR-10 dataset.

## Table of Contents
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)

## Model Architecture

The CNN consists of the following layers:

1. **Input Layer**: Receives the input image (32x32x3 for CIFAR-10).
2. **Convolutional Layers**:
   - **Conv1**: 
     - 16 filters of size 3x3.
     - Padding: 1.
   - **Conv2**: 
     - 32 filters of size 3x3.
     - Padding: 1.
   - **Conv3**: 
     - 64 filters of size 3x3.
     - Padding: 1.
3. **Max Pooling Layer**: 
   - Pooling size: 2x2.
4. **Fully Connected Layers**:
   - **FC1**: 
     - Input size: 64 * 4 * 4.
     - Output size: 500.
   - **FC2**: 
     - Input size: 500.
     - Output size: 10 (number of classes).
5. **Dropout Layers**: 
   - Dropout rate: 0.25.

### Model Code

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

## Results

The CNN achieved a test accuracy of **77%** on the CIFAR-10 dataset.

## License

This project is licensed under the MIT License.
