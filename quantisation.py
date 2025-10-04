import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class QuantizedPrunedConv2d(nn.Module):
    def __init__(self, pruned_conv_layer, n_bits=8):
        """
        Per-channel 8-bit quantization wrapper for a PrunedConv2d layer.
        """
        super().__init__()
        assert isinstance(pruned_conv_layer, PrunedConv2d), "Layer must be PrunedConv2d"

        self.orig_layer = pruned_conv_layer  # keep original for undo
        self.n_bits = n_bits
        self.is_conv = True

        # Save FP weights and bias
        self.register_buffer("weight_fp", pruned_conv_layer.weight.data.clone())
        self.register_buffer("mask", pruned_conv_layer.mask.clone())
        if pruned_conv_layer.bias is not None:
            self.register_buffer("bias_fp", pruned_conv_layer.bias.data.clone())
        else:
            self.bias_fp = None

        # Quantize only active weights (mask applied)
        masked_weight = self.weight_fp * self.mask
        self.weight_q, self.scale, self.zero_point = self.quantize_per_channel(masked_weight, n_bits)

    def quantize_per_channel(self, w, n_bits):
        C_out = w.size(0)
        q_w = torch.zeros_like(w)
        scale = torch.zeros(C_out, device=w.device)
        zero_point = torch.zeros(C_out, device=w.device)
        qmin, qmax = 0, 2**n_bits - 1

        for c in range(C_out):
            w_c = w[c]
            # Only consider active weights for min/max
            active = w_c[w_c != 0]
            if active.numel() == 0:
                # channel fully pruned → set dummy scale/zero
                scale[c] = 1.0
                zero_point[c] = 0
                q_w[c] = w_c
                continue

            w_min, w_max = active.min(), active.max()
            scale[c] = (w_max - w_min) / (qmax - qmin + 1e-8)
            zero_point[c] = qmin - w_min / (scale[c] + 1e-8)
            q_w[c] = torch.clamp(torch.round(w_c / scale[c] + zero_point[c]), qmin, qmax)

        return q_w, scale, zero_point

    def dequantize(self, q_w, scale, zero_point):
        return (q_w - zero_point.view(-1,1,1,1)) * scale.view(-1,1,1,1)

    def forward(self, x):
        # Dequantize weights
        w_deq = self.dequantize(self.weight_q, self.scale, self.zero_point)
        # Apply pruning mask
        w_deq = w_deq * self.mask
        b = self.bias_fp if self.bias_fp is not None else None

        return F.conv2d(
            x, w_deq, bias=b,
            stride=self.orig_layer.stride,
            padding=self.orig_layer.padding,
            dilation=self.orig_layer.dilation,
            groups=self.orig_layer.groups
        )

class QuantizedLinear(nn.Module):
    def __init__(self, linear_layer, n_bits=8):
        """
        Per-channel quantization wrapper for a Linear layer.
        """
        super().__init__()
        assert isinstance(linear_layer, nn.Linear), "Layer must be nn.Linear"

        self.orig_layer = linear_layer  # keep original for undo
        self.n_bits = n_bits

        # Save FP weights and bias
        self.register_buffer("weight_fp", linear_layer.weight.data.clone())
        if linear_layer.bias is not None:
            self.register_buffer("bias_fp", linear_layer.bias.data.clone())
        else:
            self.bias_fp = None

        # Quantize per output neuron (row of weight matrix)
        self.weight_q, self.scale, self.zero_point = self.quantize_per_channel(self.weight_fp, n_bits)

    def quantize_per_channel(self, w, n_bits):
        """
        w: [out_features, in_features]
        """
        C_out = w.size(0)
        q_w = torch.zeros_like(w)
        scale = torch.zeros(C_out, device=w.device)
        zero_point = torch.zeros(C_out, device=w.device)
        qmin, qmax = 0, 2**n_bits - 1

        for c in range(C_out):
            w_c = w[c]
            w_min, w_max = w_c.min(), w_c.max()
            scale[c] = (w_max - w_min) / (qmax - qmin + 1e-8)
            zero_point[c] = qmin - w_min / (scale[c] + 1e-8)
            q_w[c] = torch.clamp(torch.round(w_c / scale[c] + zero_point[c]), qmin, qmax)

        return q_w, scale, zero_point

    def dequantize(self, q_w, scale, zero_point):
        return (q_w - zero_point.view(-1,1)) * scale.view(-1,1)

    def forward(self, x):
        w_deq = self.dequantize(self.weight_q, self.scale, self.zero_point)
        b = self.bias_fp if self.bias_fp is not None else None
        return F.linear(x, w_deq, b)

def quantize_pruned_model(model, n_bits=8):
    """
    Recursively replace PrunedConv2d layers with QuantizedPrunedConv2d.
    """
    for name, module in model.named_children():
        if isinstance(module, PrunedConv2d):
            # print(module)
            setattr(model, name, QuantizedPrunedConv2d(module, n_bits))
        elif isinstance(module, nn.Linear):
            setattr(model, name, QuantizedLinear(module, n_bits))
        else:
            quantize_pruned_model(module, n_bits)
    return model

def undo_quantization_pruned(model):
    """
    Restore original PrunedConv2d layers from QuantizedPrunedConv2d wrapper.
    """
    for name, module in model.named_children():
        if isinstance(module, QuantizedPrunedConv2d):
            setattr(model, name, module.orig_layer)
        else:
            undo_quantization_pruned(module)
    return model

def apply_sensitivity_based_quant(model, layer_sensitivities, first_ratio=0.6, n_bits_first=4, n_bits_rest=8):
    sorted_layers = sorted(layer_sensitivities.items(), key=lambda x: x[1])
    n_layers = len(sorted_layers)
    split_idx = int(n_layers * first_ratio)
    
    layer_bit_dict = {}
    for i, (name, _) in enumerate(sorted_layers):
        if i < split_idx:
            layer_bit_dict[name] = n_bits_first
        else:
            layer_bit_dict[name] = n_bits_rest
    
    def replace_layers(module, string = ''):
        for name, child in module.named_children():
            string += name

            if isinstance(child, PrunedConv2d) and string in layer_bit_dict:
                setattr(module, name, QuantizedPrunedConv2d(child, n_bits=layer_bit_dict[string]))
            elif isinstance(child, nn.Linear) and string in layer_bit_dict:
                setattr(module, name, QuantizedLinear(child, n_bits=layer_bit_dict[string]))  
            else:
                string += "."
                replace_layers(child, string)
                string = string[:-1]
            l = len(name)
            string = string[:-l]
        return module

    return replace_layers(model), layer_bit_dict

def layer_sensitivities(net):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])
    calibration_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    calibration_loader = torch.utils.data.DataLoader(calibration_dataset, batch_size=64, shuffle=True)
    
    calibration_batches = 5
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    layer_sensitivities = {}
    channel_sensitivities = {}
    
    for i, (images, targets) in enumerate(calibration_loader):
        if i >= calibration_batches:
            break
        images, targets = images.to(device), targets.to(device)
        
        net.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, targets)
        loss.backward()
        
        for name, module in net.named_modules():
            # ----- Conv2d -----
            if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
                grad_abs = module.weight.grad.abs() 
                channel_sens = grad_abs.view(module.out_channels, -1).mean(dim=1) 
    
            # ----- Linear -----
            elif isinstance(module, nn.Linear) and module.weight.grad is not None:
                grad_abs = module.weight.grad.abs() 
                channel_sens = grad_abs.mean(dim=1) 
    
            else:
                continue  # skip ones without weights
    
            # Accumulate
            if name in layer_sensitivities:
                layer_sensitivities[name] += channel_sens.detach().cpu()
                
            else:
                layer_sensitivities[name] = channel_sens.detach().cpu()
    
    
    for name in layer_sensitivities:
        layer_sensitivities[name] /= calibration_batches
    
    layer_avg_sensitivities = {name: vals.mean().item() for name, vals in layer_sensitivities.items()}
    return layer_avg_sensitivities


def quantize_activation(x, num_bits=4, eps=1e-6):
    qmin, qmax = 0, (1 << num_bits) - 1

    x_clamped = torch.clamp(x, min=eps)
    min_val = x_clamped.amin(dim=(0, 2, 3), keepdim=True)
    max_val = x_clamped.amax(dim=(0, 2, 3), keepdim=True)

    log_min = torch.log(min_val)
    log_max = torch.log(max_val)

    scale = (log_max - log_min) / (qmax - qmin)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    q_x = torch.round((torch.log(x_clamped) - log_min) / scale).clamp(qmin, qmax)
    dq_x = torch.exp(q_x * scale + log_min)
    return dq_x


class QuantizedReLU6(nn.Module):
    def __init__(self, relu6_module, num_bits=4):
        """
        Wrap an existing ReLU6 module to quantize its output.
        Keep reference to original for undoing.
        """
        super().__init__()
        assert isinstance(relu6_module, nn.ReLU6)
        self.orig_module = relu6_module  # store original for undo
        self.num_bits = num_bits

    def forward(self, x):
        x = self.orig_module(x)
        x = quantize_activation(x, self.num_bits)
        return x


def replace_relu6_with_quantized(module, num_bits=4):
    """
    Recursively replace ReLU6 with QuantizedReLU6 in a module.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU6):
            setattr(module, name, QuantizedReLU6(child, num_bits=num_bits))
        else:
            replace_relu6_with_quantized(child, num_bits)


def undo_quantized_relu6(module):
    """
    Recursively restore original ReLU6 modules from QuantizedReLU6 wrappers.
    """
    for name, child in module.named_children():
        if isinstance(child, QuantizedReLU6):
            setattr(module, name, child.orig_module)
        else:
            undo_quantized_relu6(child)
