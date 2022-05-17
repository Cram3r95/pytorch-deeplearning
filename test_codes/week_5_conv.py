import nntplib
import torch
from torch import nn

# N.B. For every convolution, the kernels represent the weights!! That's why if
# we print the weight size it matches with our filter bank kernel size

## 1D convolution

# Output size: out_size = n (num_samples) - k (kernel_size) + 1
# N.B. This is the output for every vector after applying a single kernel over
# the whole 1D input. So, if we have L kernels, we will have L vectors of out_size

conv = nn.Conv1d(2, 16, 3) # 2 channels (stereo signal), 16 kernels of size 3
# In this case, since the input has 2 channels, it is like a mono image, where
# for each pixel we have its x and y position. A kernel, regardless the type
# of convolution, must have the same dimensions than the input (in this case
# 2, which is the thickness of the kernel)
print("Weight size: ", conv.weight.size()) # 16 x 2 x 3
print("Bias size: ", conv.bias.size()) # 16. The bias is a single vector of length 
# m (number of kernels = filter bank length), and it is summed up to the result
# of applying a given kernel to the input (y = Wx + b)

x = torch.rand(1,2,64) # batch size = 1, channels = 2, 64 samples
print("Size after convolution: ", conv(x).size()) # 1, 16, 62 (= 64 - 3 + 1)

conv = nn.Conv1d(2, 16, 5)
print("Size after convolution with kernel size = 5: ", conv(x).size()) # 1, 16, 60 (= 64 - 5 + 1)

print("------------------")

## 2D convolution

# Output size: feature_map_weight = in_image_width - kernel_width + 1
#              feature_map_height = in_image_height - kernel_height + 1

# As stated above with the 1D convolution, the output will be L feature maps with size
# feature_map_weight x feature_map_height, being L the number of kernels

x = torch.rand(1, 20, 64, 128) # 1 sample (batch), 20 channels (image channels, in
# case of working with RGB, this argument would be 3), height (rows) 64, and width
# (columns) 128 

conv = nn.Conv2d(20, 16, (3, 5)) # 20 channels, 16 kernels, kernel size is 3 x 5 (tuple)
print("Kernel 2D: ", conv.weight.size()) # Filter bank = Weights = torch.Size([16, 20, 3, 5])
print("Outout 2D size: ", conv(x).size()) # output: torch.Size([1 = batch_size, 16 = num_kernels, 62 = height after conv
# , 124 = width after conv])

# Stride & Padding

# We can observe that the kernel size is 3 (height) x 5 (width), so the original image
# has lost 2 pixels in the y-axis (rows) and 4 pixels in the x-axis (columns)

# Add padding (1 in the rows, that is, 1 pixel at the top and 1 at the bottom, and 2 
# in the columns). Stride = 1 means standard sliding of the kernel

# 20 channels, 16 kernels of size 3 x 5, stride is 1, padding of 1 and 2
conv = nn.Conv2d(20, 16, (3, 5), 1, (1, 2))
print("Output 2D after padding: ", conv(x).size()) # output: torch.Size([1, 16, 64, 128])

