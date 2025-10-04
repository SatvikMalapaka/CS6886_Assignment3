import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class PrunedConv2d(nn.Conv2d):
    def __init__(self, conv_layer, prune_ratio=0.4):
        super().__init__(
            in_channels=conv_layer.in_channels,
            out_channels=conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
            bias=(conv_layer.bias is not None),
            padding_mode=conv_layer.padding_mode,
        )

        # Copy weights and bias
        self.weight = nn.Parameter(conv_layer.weight.data.clone())
        if conv_layer.bias is not None:
            self.bias = nn.Parameter(conv_layer.bias.data.clone())

        # Build pruning mask
        W = self.weight.data
        out_channels = W.size(0)
        norms = W.view(out_channels, -1).abs().sum(dim=1)  # L1 norm
        num_prune = int(prune_ratio * out_channels)
        prune_indices = torch.argsort(norms)[:num_prune]

        mask = torch.ones(out_channels, dtype=torch.float32, device=W.device)
        mask[prune_indices] = 0
        self.register_buffer("mask", mask.view(-1, 1, 1, 1))

    def forward(self, x):
        pruned_weight = self.weight * self.mask
        return F.conv2d(
            x, pruned_weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )

def replace_conv_with_pruned(model, prune_ratio=0.4):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            setattr(model, name, PrunedConv2d(module, prune_ratio))
        else:
            replace_conv_with_pruned(module, prune_ratio)
    return model


def fine_tune(model, device, train_loader, test_loader, epochs=10, learn_r=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_r, momentum=0.9, weight_decay=5e-4)

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        avg_loss = running_loss / total
        _,test_acc = test(model, test_loader, device)

        print(f"Epoch {epoch+1}/{epochs} | Loss={avg_loss:.4f} | Train Acc={train_acc:.2f}% | Test Acc={test_acc:.2f}%")

    return model

def compute_sparsity(model):
    total_params = 0
    zero_params = 0

    for module in model.modules():
        if isinstance(module, PrunedConv2d):
            W = module.weight.data
            masked_W = W * module.mask  # apply pruning mask
            total_params += masked_W.numel()
            zero_params += (masked_W == 0).sum().item()

    sparsity_ratio = zero_params / total_params if total_params > 0 else 0
    print(f"Model sparsity: {sparsity_ratio*100:.2f}% "
          f"({zero_params}/{total_params} weights pruned)")
    return sparsity_ratio
