# Pixel-Pair-Encoding

1. [pixel_bpe/base.py](pixel_bpe/base.py): Implements the `Tokenizer` class, which is the base class. It contains the `train`, `encode`, and `decode` stubs, save/load functionality, and there are also a few common utility functions. This class is not meant to be used directly, but rather to be inherited from.
2. [pixel_bpe/basic.py](pixel_bpe/basic.py): Implements the `BasicTokenizer`, the simplest implementation of the BPE algorithm that runs directly on image.


## Instructions
1. run [mnist_vis.py](vis_mnist.py) to download and visualize MNIST dataset.
2. run [train.py](train.py) to train a model (vocabulary).
3. run [test.py](test.py) to inference.

## Experiment Notes
Key parameters: 
merge threshold = 2 occurences, maximum merge number for each image = 100

Training dataset: 
MNIST training dataset including 60,000 .jpg images

Testing dataset: 
The first image of MNIST testing dataset 

Device: Macbook Air M2, 2022
Training time: Training took 465.25 seconds

Result:
vocabulary size: 256 + 16,039.
During test, be able to reduce the length of the first image of MNIST testing dataset from 784 to 80.
## Reference
[1]. https://github.com/karpathy/minbpe

## Acknowledgement
Yubo Huang\
Enmao Diao