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

## Compression
In order to compress the trained file, run the `compress.py` file. The flags are as follows:
1. `--weights`: To add the path to the weights file (pth). Defaults to the weights file given.
2. `--pr`: Pruning ratio. Defualt: 0.4
3. `--finetune_epochs`: Number of epochs to fine tune the pruning process. Default: 10
4. `--n_bits_first`: Number of bits for the first chunk during weight quantisation. Default: 4
5. `--n_bits_rest`: Number of bits for the next chunk during weight quantisation. Default: 8
6. `--quant_ratio`: The first-k ratio which goes to the first chunk. Basically the ratio that belongs to the lower precision. Default: 0.6
7. `--n_bits_relu`: Number of bits for the activations. Default: 8

In order to run a uniform quantisation, ensure the `--n_bits_first` and `--n_bits_rest` are equal. Example, in order to uniformly quantise to 8 bits:
```bash
python3 compress.py --n_bits_first 8 --n_bits_rest 8
```
