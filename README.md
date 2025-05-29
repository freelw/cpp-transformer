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

```
./build_gpu.sh 
```

### perform inference using a pre-trained model

```
./test_translation.sh
```

#### output

```
./test_translation.sh 
~/project/cpp-transformer/checkpoints/save ~/project/cpp-transformer
~/project/cpp-transformer
OMP_THREADS: 8
epochs : 0
dropout : 0.2
lr : 0.001
tiny : 0
data loaded
warmUp done
parameter size = 21388
all parameters require_grad = true
loading from checkpoint : ./checkpoints/save/checkpoint_20250402_150847_40.bin
loaded from checkpoint
serving mode
go now . <eos> 
translate res : <bos> allez-y maintenant maintenant maintenant . <eos> 
i try . <eos> 
translate res : <bos> j'essaye . <eos> 
cheers ! <eos> 
translate res : <bos> santé ! <eos> 
get up . <eos> 
translate res : <bos> lève-toi . <eos> 
hug me . <eos> 
translate res : <bos> <unk> dans vos bras ! <eos> 
i know . <eos> 
translate res : <bos> je sais . <eos> 
no way ! <eos> 
translate res : <bos> en aucune manière ! <eos> 
be nice . <eos> 
translate res : <bos> soyez gentille ! <eos> 
i jumped . <eos> 
translate res : <bos> j'ai sauté . <eos> 
congratulations ! <eos> 
translate res : <bos> à ! <eos> 
```

### train with a small dataset

```
./train_tiny.sh
```

#### output

```
./train_tiny.sh 
OMP_THREADS: 8
epochs : 10
dropout : 0.2
lr : 0.001
tiny : 0
data loaded
warmUp done
parameter size = 21388
all parameters require_grad = true
[300/300]checkpoint saved : ./checkpoints/checkpoint_20250402_164906_0.bin
epoch 0 loss : 9.0757 emit_clip : 3
[300/300]epoch 1 loss : 7.90043 emit_clip : 3
[300/300]epoch 2 loss : 6.8447 emit_clip : 3
[300/300]epoch 3 loss : 5.85042 emit_clip : 3
[300/300]epoch 4 loss : 5.00354 emit_clip : 3
[300/300]epoch 5 loss : 4.38405 emit_clip : 3
[300/300]epoch 6 loss : 3.96133 emit_clip : 3
[300/300]epoch 7 loss : 3.70218 emit_clip : 3
[300/300]epoch 8 loss : 3.51153 emit_clip : 3
[300/300]checkpoint saved : ./checkpoints/checkpoint_20250402_164906_9.bin
epoch 9 loss : 3.35273 emit_clip : 3
```

## Derivation of backpropagation gradient formulas

* [Derivation](doc/equations/readme.md)

## Reference Materials

* [Dive into Deep Learning](https://d2l.ai/)
* [recognizing_handwritten_digits](https://github.com/freelw/recognizing_handwritten_digits)
* [micrograd](https://github.com/EurekaLabsAI/micrograd)
