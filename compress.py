from prune import *
from quantisation import *
from model import MobileNetV2
import argparse
from utils import *
from pathlib import Path

def get_args():
  parser = argparse.ArgumentParser(description="Compression script")
  parser.add_argument('--pr', type=float, default=0.4, help='Pruning Ratio')
  parser.add_argument('--finetune_epochs', type=int, default=10, help='Number of epochs for fine tuning')
  parser.add_argument('--n_bits_first', type=int, default=4, help='Number of bits for first half of quantisation')
  parser.add_argument('--n_bits_rest', type=int, default=8, help='Number of bits for second half of quantisation')
  parser.add_argument('--quant_ratio', type=float, default=0.6, help='Ratio of split between two bit types for quantisation')
  parser.add_argument('--n_bits_relu', type=int, default=8, help='Number of bits for activations')
  parser.add_argument('--weights', type=Path, default=Path("mobilenetv2_cifar10_baseline.pth"), help='Path to pretrained weights')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = get_args()
  prune_ratio = args.pr
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print("Using device:", device)
  net = MobileNetV2(num_classes=10)
  net = net.to(device)
  checkpoint = torch.load(args.weights, map_location=torch.device(device))
  net.load_state_dict(checkpoint)
  print("Loaded state.....")
  net = replace_conv_with_pruned(net, prune_ratio)
  _, pre_tuning_acc = test(net, test_loader, device)
  print(f"The pre tuning accuracy after pruning is {pre_tuning_acc}")
  print("Fine tuning now......")
  net = fine_tune(net, device, train_loader, args.finetune_epochs, learn_r=0.01)
  _, post_tuning_acc = test(net, test_loader, device)
  print(f"Post Pruning accuracy is {post_tuning_acc}")
  spars = compute_sparsity(net)

  #Weight Quantisation
  layer_avg_sensitivities = layer_sensitivities(net)
  net, layerwise_quant = apply_sensitivity_based_quant(net, layer_avg_sensitivities, args.quant_ratio, args.n_bits_first, args.n_bits_rest)
  _, post_weight_quant_acc = test(net, test_loader, device)
  print(f"Post Weight Quantisation accuracy is {post_weight_quant_acc}")

  #Activation Quantisation
  net = replace_relu6_with_quantized(net, args.n_bits_relu)
  _, post_act_quant_acc = test(net, test_loader, device)
  print(f"Post Activation Quantisation accuracy is {post_act_quant_acc}")

  print("\n")
  print("-------------------------SUMMARY-----------------------")
  summary = compute_model_and_weight_stats(net, layerwise_quant)
  print("Model compression ratio (orig/compressed):", summary['model_cr'])
  print("Weight compression ratio (orig/compressed):", summary['weight_cr'])
  print("Final model size (MB):", summary['final_size_mb'])
  
