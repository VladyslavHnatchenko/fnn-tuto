import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


input_size = 784  # Image size = 28*28 = 784
hidden_size = 500  # Number of nodes on hidden layer
num_classes = 10  # The number of classes at the output, from 0 to 9.
num_epochs = 5  # Number of workouts for the entire data set
batch_size = 100  # Input size for one iteration
learning_rate = 0.001  # Convergence rate


train_dataset = dsets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)


test_dataset = dsets.MNIST(
    root='./data',
    train=False,
    transform=transforms.ToTensor()
)


train_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True
)


test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


net = Net(input_size, hidden_size, num_classes)
# net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # if (i+1) % 100 == 0:  # Logging
        #     print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
        #           % (epoch+1, num_epochs, i+1, len(train_dataset) // batch_size, loss.data[0]))


correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()


print('Accuracy of the network on the 10k test images: %d %%' % (100 * correct / total))

torch.save(net.state_dict(), 'fnn_model.pkl')
