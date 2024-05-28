# epoch-V-code-guidelines

## Rearrange Over Permute

```python
from einops import rearrange
import torch

# Using rearrange from einops
x = torch.randn(32, 64, 64, 3)
# stands for batch, height, width, channels. If you work with this a lot, this starts to become a much clearer way of annotating things.
x_rearranged = rearrange(x, 'b h w c -> b c h w')

# Using permute from PyTorch
x_transposed = x.permute(0, 3, 1, 2)
```

## functional over nn.Module

## Functional Over Object-Oriented for Activation Functions and loss functions

Activation functions don't have any state, it's nice to indicate that by using the functional approach.

```python
import torch
import torch.nn.functional as F

# Using functional approach
x = torch.randn(10, 10)
x_relu = F.relu(x)

# Using object-oriented approach
relu = torch.nn.ReLU()
x_relu_obj = relu(x)
```


## Avoid Unnecessary Documentation

```python
# Unnecessary documentation
def add(a, b):
    """
    Adds two numbers together.
    
    Parameters:
    a (int): The first number
    b (int): The second number
    
    Returns:
    int: The sum of a and b
    """
    return a + b

# Clean and self-explanatory code without unnecessary docs
def add(a, b):
    return a + b
```
