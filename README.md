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
4. num_steps has been increased from 9 to 32
5. Closer Implementation to Tensor Semantics in DL2 Chapter 11
6. Enhanced Test Cases

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

### training
#### use a small corpus file
```
$ time ./transformer -e 100 -f ./resources/fra_tiny.txt 
corpus : ./resources/fra_tiny.txt
epochs : 100
batch_size : 128
gpu : 1
learning rate : 0.001
checkpoint : 
enc_vocab_size : 7939
dec_vocab_size : 13387
bos_id : 3
eos_id : 1
src_pad_id : 0
tgt_pad_id : 0
predicting : false
batch_size : 128
epoch 0 :  [384/384]loss : 9.21267
epoch 1 :  [384/384]loss : 8.16567
epoch 2 :  [384/384]loss : 7.29368
epoch 3 :  [384/384]loss : 6.52666
...
epoch 97 :  [384/384]loss : 0.211998
epoch 98 :  [384/384]loss : 0.214285
epoch 99 :  [384/384]loss : 0.2198
checkpoint saved : ./checkpoints/checkpoint_20250529_181410_99.bin

real    2m59.602s
user    2m58.975s
sys     0m0.605s
```
As shown above, training for 100 epochs on a small dataset takes approximately 3 minutes.
If you want to use the full corpus, remove the -f parameter.

### inference
Perform translation inference using the checkpoint file generated earlier.
The data will be read from the test.txt file.
```
$ ./transformer -e 0 -c ./checkpoints/checkpoint_20250529_181410_99.bin
corpus : ./resources/fra_preprocessed.txt
epochs : 0
batch_size : 128
gpu : 1
learning rate : 0.001
checkpoint : ./checkpoints/checkpoint_20250529_181410_99.bin
enc_vocab_size : 7939
dec_vocab_size : 13387
bos_id : 3
eos_id : 1
src_pad_id : 0
tgt_pad_id : 0
predicting : true
batch_size : 1
loading from checkpoint : ./checkpoints/checkpoint_20250529_181410_99.bin
loaded from checkpoint
serving mode
test file : ./test.txt
go now . -> vas-y maintenant . 
i know that it is highly unlikely that you'd ever want to go out -> je sais qu'il est hautement improbable que tu veuilles jamais sortir avec moi , mais j'ai tout de même besoin de demander au moins une fois . 
good job -> bon boulot . 
how nice ! -> comme c'est chouette ! 
```

## legacy version

[v1](https://github.com/freelw/cpp-transformer/tree/v1_freeze_20250529)


## Derivation of backpropagation gradient formulas

* [Derivation](doc/equations/readme.md)

## Reference Materials

* [Dive into Deep Learning](https://d2l.ai/)
* [recognizing_handwritten_digits](https://github.com/freelw/recognizing_handwritten_digits)
* [micrograd](https://github.com/EurekaLabsAI/micrograd)
