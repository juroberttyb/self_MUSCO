import arch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def prepare_loader(batch_size):
    train_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.RandomHorizontalFlip(),
                 transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))])

    val_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    return train_loader, val_loader

def plot_loss(train_loss, val_loss):

    plt.plot(train_loss, 'b')
    plt.plot(val_loss, 'r')
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('loss.png')

def validation(val_loader, criterion):
    correct = 0
    total = 0
    val_loss = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images, labels
            # calculate outputs by running images through the network
            outputs = net(images)
            val_loss = val_loss + criterion(outputs, labels)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    val_loss = val_loss / 10000

    return val_loss

if __name__ == "__main__":

    batch_size = 100

    train_loader, val_loader = prepare_loader(batch_size)

    net = arch.Net() #.cuda()

    from torchsummary import summary
    summary(net, input_size=(3, 32, 32))

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss() #.cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    epoch = 1
    while True:
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs, labels #.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('[%d %d] loss: %.3f' %
                    (epoch, i, loss.item()))
        
        train_loss.append(loss.item() / batch_size)
        val_loss.append(validation(val_loader, criterion))

        plot_loss(train_loss, val_loss)

        PATH = 'baseline.pth'
        torch.save(net.state_dict(), PATH)

        epoch = epoch + 1

    # plot_loss(train_loss, val_loss)

