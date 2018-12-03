# Sequence_to_sequence in Pytorch

Sequence-to-sequence neural network with attention. You can play with a toy dataset to test different configurations.
The toy dataset consists of batched (input, target) pairs, where the target is the reversed input.

## Getting Started


### Prerequisites

Install the packages with pip

```
pip install -r requirement.txt
```

### Train model

Train and evaluate models with

```
python main.py --config=<json_config_file>
```
Examples of config files are given in the "experiments" folder. All config files have to be placed in this directory.
### Hyper-parameters

You can tune the following parameters:

* decoder type (with or without Attention)
* encoder type (with or without downsampling, with or without preprocessing layers)
* the encoder's hidden dimension
* the number of recurrent layers in the encoder
* the encoder dropout
* the bidirectionality of the encoder
* the decoder's hidden dimension
* the number of recurrent layers in the decoder
* the decoder dropout
* the bidirectionality of the decoder
* batch size
* the type of attention used
etc...

