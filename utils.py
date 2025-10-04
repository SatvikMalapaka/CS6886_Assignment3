from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np

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

def compute_model_size(model, bitwidth_map, include_mask=True, activation_bits=8):
    orig_weight_bits = 0
    compressed_weight_bits = 0
    overhead_bits = 0

    for name, module in model.named_modules():
        if isinstance(module, (QuantizedPrunedConv2d, QuantizedLinear)):
            # original FP32 weights
            W = module.weight_fp.data
            orig_weight_bits += W.numel() * 32

            if hasattr(mod, "mask"):
                mask = mod.mask.detach().cpu()
                # broadcast mask to same numel as weight if necessary
                try:
                    active = (mask != 0).sum().item()
                except Exception:
                    active = (mask.view(-1) != 0).sum().item()
                compressed_weight_bits = active * quant_bits_map.get(name, default_bits)
            else:
                compressed_weight_bits = w_num * quant_bits_map.get(name, default_bits)

            comp_bits += compressed_weight_bits

            if module.bias_fp is not None:
                overhead_bits += module.bias.numel() * 32

            # quantization parameters: scale + zero-point (per channel)
            if isinstance(module, QuantizedPrunedConv2d):
                n_channels = module.out_channels
            else:
                n_channels = module.out_features
            overhead_bits += n_channels * 2 * 32  # scale + zero-point

        elif isinstance(module, nn.BatchNorm2d):
            # gamma, beta, running_mean, running_var → 4 params per channel
            n_channels = module.num_features
            overhead_bits += n_channels * 4 * 32

    # Final sizes in MB
    orig_MB = orig_weight_bits / 8 / (1024**2)
    compressed_MB = (compressed_weight_bits + overhead_bits) / 8 / (1024**2)

    # Compression ratios
    compression_weights = orig_weight_bits / compressed_weight_bits
    compression_total = orig_weight_bits / (compressed_weight_bits + overhead_bits)

    return {
        "Original size (MB)": orig_MB,
        "Compressed size (MB)": compressed_MB,
        "Weight compression ratio": compression_weights,
        "Final compression ratio (with overheads)": compression_total,
        "Overhead bits": overhead_bits,
        "Activation quant params overhead bits": act_overhead_bits
    }
