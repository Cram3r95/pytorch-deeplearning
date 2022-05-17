import torch # PyTorch framework
from torch import nn # Neural Networks sublibrary

image = torch.randn(3,10,20) # channels, height, width
d0 = image.nelement() # Total dimensions = 3 x 10 x 20 = 600

print("d0: ", d0)

class mynet(nn.Module): # Most networks in PyTorch are
                        # based on nn.Module as parent
                        # class
    def __init__(self, d0, d1, d2, d3):
        super().__init__() # Initialize the parent class
                           # (in this case nn.Module)
        self.m0 = nn.Linear(d0,d1)
        self.m1 = nn.Linear(d1,d2)
        self.m2 = nn.Linear(d2,d3)

        # Each Linear (in this case) must have as input
        # dimension the output dimension of the previous
        # one. Note that the first one takes as input
        # 3 x 10 x 20 (that is, all the pixels of our
        # "image")

    def forward(self,x): 
        # The forward method of a neural network class
        # represents what you are going to do with the 
        # input until obtaining the output

        z0 = x.view(-1) # flatten input tensor
                        # (from n,m,r -> nxmxr)
                        # (1-dimensional vector)
        s1 = self.m0(z0) # s1 represents the output of 
                         # the first Linear
        z1 = torch.relu(s1) # Apply non-linearity
        s2 = self.m1(z1)
        z2 = torch.relu(s2)
        s3 = self.m2(z2)

        return s3

if __name__ == "__main__": 
    # Only execute the code below
    # if this script is directly run (not being called by
    # another script/function)

    model = mynet(d0, 60, 40, 10)
    out = model(image)

    print("out: ", out.shape)
