import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

EPOCHS_TO_TRAIN = 50000
loss = 0  # Init Loss


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  # Init nn.Module()

        self.fc1 = nn.Linear(2, 3, True)  # Create linear layer ( Input: (x, y))
        self.fc2 = nn.Linear(3, 1, True)  # Create linear layer ( Output: (1, 0))

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))  # Sigmoid activation function
        x = self.fc2(x)
        return x


net = Net()  # Init net

# Training data

inputs = list(map(lambda s: Variable(torch.Tensor([s])), [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]))
targets = list(map(lambda s: Variable(torch.Tensor([s])), [
    [0],
    [1],
    [1],
    [0]
]))


criterion = nn.MSELoss()  # Init loss function
optimizer = optim.SGD(net.parameters(), lr=0.01)  # Init training algorithm

print("Training loop:")
for idx in range(0, EPOCHS_TO_TRAIN):
    for input, target in zip(inputs, targets):
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(input)  # Passes input through net
        loss = criterion(output, target)  # Computes loss
        loss.backward()  # Updates gradients according to loss
        optimizer.step()  # Propagates updated gradients
    if idx % 5000 == 0:
        print("Epoch "+str(idx)+" Loss: "+str(loss))

print()
print("Final results:")
for input, target in zip(inputs, targets):
    output = net(input)
    print("Input:[{},{}] Target:[{}] Predicted:[{}] Error:[{}]".format(
        int(input.data.numpy()[0][0]),
        int(input.data.numpy()[0][1]),
        int(target.data.numpy()[0]),
        round(float(output.data.numpy()[0]), 4),
        round(float(abs(target.data.numpy()[0] - output.data.numpy()[0])), 4)
    ))