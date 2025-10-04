from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import MobileNetV2
import argparse
import matplotlib.pyplot as plt

def get_args():
  parser = argparse.ArgumentParser(description="Training script")
  parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
  parser.add_argument('--plot', action='store_true', help='Whether to plot learning curves')
  args = parser.parse_args()
  return args

def lr_lambda(epoch):
  if epoch < 50:
    return 1.0      # 0.1 (initial)
  elif epoch < 75:
    return 0.1      # 0.01
  else:
    return 0.001    # 0.0001

def plotting(num_epochs, train_losses, train_accs, test_accs):
  plt.figure(figsize=(10, 4))
  plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training Loss')
  plt.legend()
  plt.grid(True)
  plt.show()
  
  plt.figure(figsize=(10, 4))
  plt.plot(range(1, num_epochs+1), train_accs, label='Train Accuracy')
  plt.plot(range(1, num_epochs+1), test_accs, label='Test Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy (%)')
  plt.title('Training & Testing Accuracy')
  plt.legend()
  plt.grid(True)
  plt.show()


if __name__ == '__main__':
  args = get_args()
  epochs = args.epochs
  print(f"Training for {epochs} epochs...")
  
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print("Using device:", device)
  net = MobileNetV2(num_classes=10)
  net.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(
    net.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=4e-5
  )

  
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
  best_acc = 0

  train_losses, train_accs = [], []
  test_losses, test_accs = [], []
  
  epoch_bar = tqdm(range(1, epochs + 1), desc="Training Progress")
  
  for epoch in epoch_bar:
    tr_loss, tr_acc = train(net, epoch, train_loader, optimizer, criterion, device)
    te_loss, te_acc = test(net, test_loader, device, criterion, epoch)

    train_losses.append(tr_loss)
    train_accs.append(tr_acc)
    test_accs.append(te_acc)
    test_losses.append(te_loss)

    epoch_bar.set_postfix(train_acc=tr_acc, test_acc=te_acc)

    if te_acc > best_acc:
      best_acc = te_acc
      torch.save(net.state_dict(), "mobilenetv2_cifar10_baseline.pth")

    scheduler.step()
  
  print("Best Test Accuracy:", best_acc)
  if args.plot():
    plotting(epochs, train_losses, train_accs, test_accs)

  
