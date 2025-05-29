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
V2 - [2025-05-29]​
1. Redesigned Tensor Class​
The tensor class has been completely redesigned to minimize memory allocation and copying operations. This optimization significantly improves memory efficiency and reduces overhead during tensor manipulation, leading to enhanced performance and faster execution times.​
​
​
2. Redesigned Backend Ops Interface​
The backend_ops interface has been revamped to support both CPU and GPU backends simultaneously. This unified interface simplifies the codebase, allowing for seamless switching between different hardware backends and providing users with the flexibility to choose the most suitable computing resource for their tasks.​
​
​
3. Redesigned Computation Flow​
a. Pre - computed Tensor Dependency Logic and Batch Memory Allocation​
The computation flow now pre - calculates the tensor dependency logic. By doing so, it enables batch memory allocation, which optimizes memory usage and reduces the time spent on individual memory allocation calls, resulting in a more efficient overall process.​
b. Compact Memory Layout​
A more compact memory layout has been implemented. This layout arranges data in a more space - efficient manner, reducing memory fragmentation and improving data access speed, thus enhancing the performance of tensor - based computations.​
c. Efficient zero_grad Implementation​
An efficient implementation of the zero_grad function has been introduced. This new implementation clears the gradients of tensors in a more optimized way, improving the performance of gradient - based optimization algorithms and speeding up the training process.​
​
​
4. Increased num_steps​
The value of num_steps has been increased from 9 to 32. This expansion allows for longer sequences or more iterative processes in relevant algorithms, providing greater flexibility and enabling more complex computations.​
​
​
5. Closer Implementation to Tensor Semantics in DL2 Chapter 11​
a. Attention - related Improvements​
transpose_qkv: An optimized implementation of transpose_qkv has been added, improving the performance of attention mechanisms that rely on this operation.​
transpose_output: The transpose_output operation has been refined to better align with the tensor semantics specified in DL2 Chapter 11, enhancing the overall accuracy and efficiency of attention - based models.​
permute: The permute operation has been updated to follow the semantic requirements more closely, enabling more precise control over tensor dimensions in attention - related computations.​
c. Cross - Entropy in Non - reduction Mode​
A new implementation of cross - entropy loss in non - reduction mode has been added. This mode provides more detailed information about the loss for each individual data point, which is useful for debugging and more fine - grained analysis.​
d. Improved Embedding Implementation​
The embedding implementation has been enhanced to better capture the semantic relationships in the data. This improvement leads to more accurate representations of input data, benefiting various natural language processing and machine learning tasks that rely on embeddings.​
​
​
6. Enhanced Test Cases​
The test suite has been strengthened with more comprehensive and robust test cases. These new test cases cover a wider range of scenarios and edge cases, ensuring the stability, reliability, and correctness of the updated features and the overall system.

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
