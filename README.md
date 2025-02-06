## Overview

1D ResNet is a deep learning model based on the classic ResNet architecture but adapted for one-dimensional data. It is designed for tasks such as time-series classification, signal processing, and other sequence-based applications.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/1D-ResNet-50.git
cd 1D-ResNet-50
# Install dependencies
pip install -r requirements.txt

```
## Usage
```python
from resnet_model import resnet
import torch

x = torch.randn(10, 5000, 12)  # Example input tensor
model = resnet(input_channels=12, num_classes=5, depth=101)
logits = model(x)
```
