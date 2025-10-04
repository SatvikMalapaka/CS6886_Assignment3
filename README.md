# CS6886_Assignment3
Assignment for CS6886: Systems Engineering for DL. Compression of MobileNetV2 for CIFAR10 dataset.

## Training
In order to train the MobileNetV2, run the `train.py` file. There are two optional flags:
1. `--epochs` can be used to vary the number of epochs used in training. Default value: 100. Usage example: `--epochs 100`
2. `--plot` is a flag that can be used to plot the training error as well as the accuracy plots.

Therefore, to run the `train.py` file with 100 epochs and plotting enabled,
```bash
python3 train.py --epochs 100 --plot
```
