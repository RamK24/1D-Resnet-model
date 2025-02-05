<h2>Overview</h2>
1D ResNet-50 is a deep learning model based on the classic ResNet-50 architecture but adapted for one-dimensional data. It is designed for tasks such as time-series classification, signal processing, and other sequence-based applications.

<h2>Installation</h2>
```
# Clone the repository
git clone https://github.com/yourusername/1D-ResNet-50.git
cd 1D-ResNet-50

# Install dependencies
pip install -r requirements.txt

```

<h2>Usage</h2>
```
from resnet_model import resnet
import torch
x = torch.randn(10, 5000, 12)
model = resnet(12, 5, 101)
logits = model(x)
```
