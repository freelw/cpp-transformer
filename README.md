# cpp-transformer
A C++ implementation of Transformer without special library dependencies, including training and inference.

This project replicates the content of [Chapter 11](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html) on Transformers in Dive into Deep Learning. It builds an English-French machine translation model using C++. The project develops its own automatic differentiation framework and only depends on the C++ standard library, aiming to help users understand the underlying principles of Transformers.

## Project Highlights
### Principle - Oriented

We construct the model starting from fundamental operations without relying on deep learning frameworks. This approach clearly demonstrates the operational mechanism of Transformers.

### Automatic Differentiation

Our self - developed automatic differentiation framework simplifies the gradient calculation process, facilitating a better understanding of the backpropagation algorithm.

### Low Dependencies

The project only depends on the C++ standard library. While its performance may not be as high - end as those with advanced libraries, it clearly showcases every computational detail. This characteristic allows users to gain a profound understanding of the backpropagation algorithm and the underlying principles of the Transformer architecture.

## Update Log
V2 - [2025-05-29]
1. Redesigned Tensor Class
2. Redesigned Backend Ops Interface
3. Redesigned Computation Flow
    *  Pre - computed Tensor Dependency Logic and Batch Memory Allocation
    * Compact Memory Layout
    * Efficient zero_grad Implementation
4. Closer Implementation to Tensor Semantics in DL2 Chapter 11
5. Enhanced Test Cases

## Quick start

### build

#### for gpu
```
./build_gpu.sh 
```
The program compiled in this way supports both CPU and GPU. You can use the -g parameter to switch between them.

#### for cpu
```
./build_cpu.sh
```
If you don't have a CUDA environment, you can also try the CPU version. Note that this version is extremely slow and is only intended for comparing and verifying the correctness of the GPU version.

#### for mac
```
./build_mac_cpu.sh
```

### training
Align the training data volume (512 pairs) of Chapter 11 Transformer in d2l.
```
$ time ./transformer -e 30
corpus : ./resources/fra_preprocessed_512.txt
epochs : 30
batch_size : 128
gpu : 1
learning rate : 0.001
checkpoint :
enc_vocab_size : 195
dec_vocab_size : 214
bos_id : 3
eos_id : 1
src_pad_id : 0
tgt_pad_id : 0
predicting : false
batch_size : 128
epoch 0 :  [512/512]loss : 4.62015
epoch 1 :  [512/512]loss : 3.39543
epoch 2 :  [512/512]loss : 2.96776
epoch 3 :  [512/512]loss : 2.45226
epoch 4 :  [512/512]loss : 2.20506
epoch 5 :  [512/512]loss : 1.94157
epoch 6 :  [512/512]loss : 1.76016
epoch 7 :  [512/512]loss : 1.58783
epoch 8 :  [512/512]loss : 1.46
epoch 9 :  [512/512]loss : 1.35267
epoch 10 :  [512/512]loss : 1.23456
epoch 11 :  [512/512]loss : 1.11818
epoch 12 :  [512/512]loss : 1.02721
epoch 13 :  [512/512]loss : 0.930991
epoch 14 :  [512/512]loss : 0.868043
epoch 15 :  [512/512]loss : 0.797028
epoch 16 :  [512/512]loss : 0.730525
epoch 17 :  [512/512]loss : 0.685426
epoch 18 :  [512/512]loss : 0.670126
epoch 19 :  [512/512]loss : 0.635286
epoch 20 :  [512/512]loss : 0.580065
epoch 21 :  [512/512]loss : 0.558903
epoch 22 :  [512/512]loss : 0.528207
epoch 23 :  [512/512]loss : 0.49648
epoch 24 :  [512/512]loss : 0.482626
epoch 25 :  [512/512]loss : 0.456417
epoch 26 :  [512/512]loss : 0.452462
epoch 27 :  [512/512]loss : 0.432102
epoch 28 :  [512/512]loss : 0.408004
epoch 29 :  [512/512]loss : 0.395327
checkpoint saved : ./checkpoints/checkpoint_20250603_111836_29.bin

real    0m44.835s
user    0m44.531s
sys     0m0.272s

```

### inference
Perform translation inference using the checkpoint file generated earlier.
The data will be read from the test.txt file.
```
$ ./transformer -e 0 -c ./checkpoints/checkpoint_20250603_111836_29.bin
corpus : ./resources/fra_preprocessed_512.txt
epochs : 0
batch_size : 128
gpu : 1
learning rate : 0.001
checkpoint : ./checkpoints/checkpoint_20250603_111836_29.bin
enc_vocab_size : 195
dec_vocab_size : 214
bos_id : 3
eos_id : 1
src_pad_id : 0
tgt_pad_id : 0
predicting : true
batch_size : 1
loading from checkpoint : ./checkpoints/checkpoint_20250603_111836_29.bin
loaded from checkpoint
serving mode
test file : ./test.txt
go . -> va .
i lost . -> j'ai perdu .
he's calm . -> il est mouillé .
i'm home . -> je suis chez moi .
```

## legacy version

[v1](https://github.com/freelw/cpp-transformer/tree/v1_freeze_20250529)


## Derivation of backpropagation gradient formulas

* [Derivation](doc/equations/readme.md)

## Reference Materials

* [Dive into Deep Learning](https://d2l.ai/)
* [recognizing_handwritten_digits](https://github.com/freelw/recognizing_handwritten_digits)
* [micrograd](https://github.com/EurekaLabsAI/micrograd)
