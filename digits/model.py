import torch
import torch.nn.functional as fn
import main

class Net(torch.nn.Module):

    # Initiate model layers
    def __init__(self):
        super().__init__()

        # Convolutional layers
        # Params:
        #   - input channels/dimensions: RGB channels of the input image
        #   - output channels/dimensions: Keeps all the colour channels 
        #   - kernel size: size of the window that 'convolutes' the image
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
        self.conv3 = torch.nn.Conv2d(3, 1, 3)

        # Dense/fully connected layers
        self.dense1 = torch.nn.Linear(10, 10)
        self.dense2 = torch.nn.Linear(10, 1)

        # Calculate loss with 'Negative-Log-Likelihood-Loss'
        # loss(x, y) = -(log(y))
        self.loss_fn = torch.nn.NLLLoss()
        # Optimise network with backpropogation
        self.optimiser = torch.optim.AdamW(self.parameters(), lr=0.0001)

    def forward(self, x):
        # Pass data through convolutional layers
        x = fn.max_pool2d(fn.relu(self.conv1(x)), (2, 2))
        x = fn.max_pool2d(fn.relu(self.conv2(x)), (2, 2))
        x = fn.max_pool2d(fn.relu(self.conv3(x)), (2, 2))

        # Left with 10x10 neuron array on the cross-section
        print("After convs:", x.shape)

        # Pass data through the fully connected layers
        x = fn.relu(self.dense1(x))
        x = self.dense2(x)
        # Softmax activation on the output to give a probability ratio
        return fn.log_softmax(x, dim=x)

    def optimise(self, x, y):
        self.zero_grad()
        loss = self.loss_fn(x, y)
        print("Loss:", loss)
        loss.backward()
        self.optimiser.step()

net = Net()
x, y = main.get_batch(1)
y = torch.Tensor(y).view([1, 10])
output = net(torch.Tensor(x).view([1, 3, 100, 100])).view([1, 1, 10])
net.zero_grad()
loss = net.loss_fn(output, y)

#net.optimise(output, y)
