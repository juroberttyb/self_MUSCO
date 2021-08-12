from numpy.core.fromnumeric import resize
import arch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import res_arch as res

batch_size, epoch = 80, 150

def prepare_loader(batch_size):
    train_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    val_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize(224),
                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)

    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                            shuffle=False, num_workers=4)

    return train_loader, val_loader

def plot_loss(train_loss, val_loss):

    plt.plot(train_loss, 'b')
    plt.plot(val_loss, 'r')
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('loss.png')

def validation(net, val_loader, criterion, fast=False):
    correct = 0
    total = 0
    val_loss = 0
    i = 0
    fast_num = 3
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            # calculate outputs by running images through the network
            outputs = net(images)
            val_loss = val_loss + criterion(outputs, labels)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i+1 == fast_num and fast:
                # print(val_loss.item() / (batch_size * fast_num))
                return val_loss.item() / (batch_size * fast_num)
            i = i+1

    print('Accuracy of the network on the 10000 test images: %.2f %%' % (
        100 * correct / total))

    val_loss = val_loss / 10000.

    return val_loss.item()

def train(net, batch_size, epoch, criterion, optimizer, train_loader, val_loader, model_path):
    train_loss, val_loss = [], []
    for k in range(epoch):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda() #.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print('[%d %d] loss: %.3f' %
                    (k, i, loss.item()))
        
        train_loss.append(loss.item() / batch_size)
        val_loss.append(validation(net, val_loader, criterion))

        plot_loss(train_loss, val_loss)

        torch.save(net.state_dict(), model_path)

if __name__ == "__main__":
    train_loader, val_loader = prepare_loader(batch_size)

    net = res.resnet50().cuda() # arch.Net()

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train(net, batch_size, epoch, criterion, optimizer, train_loader, val_loader, "baseline.pth")
