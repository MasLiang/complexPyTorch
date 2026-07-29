# complexPyTorch

A high-level toolbox for using complex valued neural networks in PyTorch.

## Current Training Route

The active research route now exposes:

- Phase 1: full-precision complex Bi-Real ResNet.
- Phase 2: binary complex activations and binary complex convolution weights.
- Phase 3: LUT-as-neuron convolutions. Each 4-input LUT consumes two binary
  complex activations and emits one real or imaginary output bit.

Run them with:

```bash
GPU_ID=0 WORKDIR=runs/phase1 ./run_phase1.sh
GPU_ID=1 WORKDIR=runs/phase2 \
  CHECKPOINT=bi_workdir/chkpts/Bestmodel_phase1.pt \
  ./run_phase2.sh
GPU_ID=2 WORKDIR=runs/phase3_pair_lut \
  CHECKPOINT=bi_workdir/chkpts/Bestmodel_phase2.pt \
  ./run_phase3.sh
```

Phase 3 can also initialize the entire network and all LUT logits randomly:

```bash
GPU_ID=2 TRAIN_FROM_SCRATCH=1 LUT_LOGIT_INIT=1.0 LR=0.01 \
  WORKDIR=runs/phase3_pair_lut_scratch ./run_phase3.sh
```

For a controlled comparison with the real-valued LUT-BiReal implementation,
use the dedicated recipe:

```bash
GPU_ID=1 WORKDIR=runs/phase3_pair_lut_real_compatible \
  ./run_phase3_real_compatible.sh
```

This keeps the complex 4-input/2-output pair-LUT architecture, but matches the
real implementation's bimodal logits, hard-forward identity STE, Adam,
unclipped gradients, linear learning-rate decay, CIFAR-10 AutoAugment,
normalization, label smoothing, and 50k-image training split. It also defaults
to the packed binary CUDA LUT kernel used for hard table lookup; the activation
addresses are encoded as `0/1` probabilities and thresholded at `0.5`, matching
the real-domain Bi-Real input-gradient convention. Since that split uses test
accuracy for model selection, it is intended as a training-mechanism comparison
rather than an unbiased final test report.

`LR`, `BATCH_SIZE`, `NUM_EPOCHS`, and `SCHEDULE` are explicit launcher
inputs and are never silently replaced. Phase-specific best and last
checkpoints are saved under each workdir's `chkpts/` directory.

The complete active-route contract is in
[CURRENT_TECHNICAL_ROUTE.md](CURRENT_TECHNICAL_ROUTE.md). All Phase 2.1+,
LUT5/LUT6, C8, MLP, magnitude, dominance, and fixed-analytic experiments
were preserved under
[`backup/route_reset_phase12_20260729/`](backup/route_reset_phase12_20260729/).
The generic LUT complex layers and CUDA backends remain available in the
active source tree. Phase 1/2 do not instantiate LUTs; Phase 3 uses only the
new pair-LUT neuron path.

Before version 1.7 of PyTroch, complex tensor were not supported. 
The initial version of **complexPyTorch** represented complex tensor using two tensors, one for the real and one for the imaginary part.
Since version 1.7, compex tensors of type `torch.complex64` are allowed, but only a limited number of operation are supported.
The current version **complexPyTorch** use complex tensors (hence requires PyTorch version >= 1.7) and add support for various operations and layers.

## Installation
```bash
pip install complexPyTorch
```

## Complex Valued Networks with PyTorch

Artificial neural networks are mainly used for treating data encoded in real values, such as digitized images or sounds. 
In such systems, using complex-valued tensors would be quite useless. 
However, for physic related topics, in particular when dealing with wave propagation, using complex values is interesting as the physics typically has linear, hence more simple, behavior when considering complex fields. 
complexPyTorch is a simple implementation of complex-valued functions and modules using the high-level API of PyTorch. 
Following [[C. Trabelsi et al., International Conference on Learning Representations, (2018)](https://openreview.net/forum?id=H1T2hmZAb)], it allows the following layers and functions to be used with complex values:
* Linear
* Conv2d
* ConvTranspose2d
* MaxPool2d
* AvgPool2d
* Relu (&#8450;Relu)
* Sigmoid
* Tanh
* Dropout2d
* BatchNorm1d (Naive and Covariance approach)
* BatchNorm2d (Naive and Covariance approach)
* GRU/BN-GRU Cell

## Citating the code

If the code was helpful to your work, please consider citing it:

[![DOI](https://img.shields.io/badge/DOI-10.1103%2FPhysRevX.11.021060-blue)](https://doi.org/10.1103/PhysRevX.11.021060)


## Syntax and usage

The syntax is supposed to copy the one of the standard real functions and modules from PyTorch. 
The names are the same as in `nn.modules` and `nn.functional` except that they start with `Complex` for Modules, e.g. `ComplexRelu`, `ComplexMaxPool2d` or `complex_` for functions, e.g. `complex_relu`, `complex_max_pool2d`.
The only usage difference is that the forward function takes two tensors, corresponding to real and imaginary parts, and returns two ones too.

## BatchNorm

For all other layers, using the recommendation of [[C. Trabelsi et al., International Conference on Learning Representations, (2018)](https://openreview.net/forum?id=H1T2hmZAb)], the calculation can be done in a straightforward manner using functions and modules form `nn.modules` and `nn.functional`. 
For instance, the function `complex_relu` in `complexFunctions`, or its associated module `ComplexRelu` in `complexLayers`, simply performs `relu` both on the real and imaginary part and returns the two tensors.
The complex BatchNorm proposed in [[C. Trabelsi et al., International Conference on Learning Representations, (2018)](https://openreview.net/forum?id=H1T2hmZAb)] requires the calculation of the inverse square root of the covariance matrix.
This is implemented in `ComplexbatchNorm1D` and `ComplexbatchNorm2D` but using the high-level PyTorch API, which is quite slow.
The gain of using this approach, however, can be experimentally marginal compared to the naive approach which consists in simply performing the BatchNorm on both the real and imaginary part, which is available using `NaiveComplexbatchNorm1D` or `NaiveComplexbatchNorm2D`.


## Example

For illustration, here is a small example of a complex model.
Note that in that example, complex values are not particularly useful, it just shows how one can handle complex ANNs.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d

batch_size = 64
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = datasets.MNIST('../data', train=True, transform=trans, download=True)
test_set = datasets.MNIST('../data', train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size= batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size= batch_size, shuffle=True)

class ComplexNet(nn.Module):
    
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(1, 10, 5, 1)
        self.bn  = ComplexBatchNorm2d(10)
        self.conv2 = ComplexConv2d(10, 20, 5, 1)
        self.fc1 = ComplexLinear(4*4*20, 500)
        self.fc2 = ComplexLinear(500, 10)
             
    def forward(self,x):
        x = self.conv1(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        x = self.bn(x)
        x = self.conv2(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        x = x.view(-1,4*4*20)
        x = self.fc1(x)
        x = complex_relu(x)
        x = self.fc2(x)
        x = x.abs()
        x =  F.log_softmax(x, dim=1)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ComplexNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).type(torch.complex64), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                loss.item())
            )

# Run training on 50 epochs
for epoch in range(50):
    train(model, device, train_loader, optimizer, epoch)
```
       

## Acknowledgments

I want to thank Piotr Bialecki for his invaluable help on the PyTorch forum.
# LUT-Complex-NN
