import torch
from torch import nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        # N.B. If you need to have layers with different
        # parameter vector (weights) that must be learnt
        # separately, you need to specify a different
        # layer per each one

        # On the other hand, layers that do not have
        # parameter vector (activation function, pooling
        # functions etc.) they are specified in F

        # import torch.nn.functional as F

        # Conv2D(input_size (depth),output_size (depth = feature maps),
        # kernel_size,stride)
        self.conv1 = F.Conv2D(1,20,5,1)
        self.conv2 = F.Conv2D(20,20,5,1) # Now the input
        # to this layer has a depth of 20 (20 features map)

        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x): # x = input
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2) # input, kernel_size, stride
        # Since we only specify a 2 for the kernel_size,
        # it is assumed to be squared
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50) # The 2nd dimension
        # is specified in order to match with the input
        # size of the first linear
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        output = F.logsoftmax(x, dim=1) # 10 scores (output
        # size of Linear2), that all of them sum up = 1

        return output 