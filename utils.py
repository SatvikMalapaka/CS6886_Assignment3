from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def train(net, epoch, train_loader, optimizer, criterion, device = device):
    net.train()
    running_loss, correct, total = 0, 0, 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(train_loader)
    acc = 100. * correct / total
    print(f"Epoch {epoch} | Train Loss: {avg_loss:.3f} | Train Acc: {acc:.2f}%")
    return avg_loss, acc


def test(net, test_loader, device = device, criterion = None, epoch=None):
    net.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if epoch is not None:
              loss = criterion(outputs, targets)
              test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = 0
    acc = 100. * correct / total
    if epoch is not None:
      avg_loss = test_loss / len(test_loader)
      print(f"Epoch {epoch} | Test  Loss: {avg_loss:.3f} | Test Acc: {acc:.2f}%")
    else:
      print(f"Test Acc: {acc:.2f}%")
    return avg_loss, acc
