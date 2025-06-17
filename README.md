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

V2.01 - [2025-06-08]
1. Supported a simple language model.

V2.02 - [2025-06-14]
1. Supported Metal

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

#### for mac gpu

Metal is now supported, and the GPU on Mac can be used now.

My MacBook hardware and software information
* Chip : Apple M1
* OS Version : 15.5 (24F74)

```
./build_mac_gpu.sh
```


#### for mac cpu
```
./build_mac_cpu.sh
```

## Translation

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

## Language Model

A language model built with a two-layer decoder, trained on the first 256 tokens from timemachine_preprocessed.txt, reads the text starting from test_lm.txt during inference.

### training
```
$ ./lm -e 10 -m 256
corpus : ./resources/time_machine/timemachine_preprocessed.txt
epochs : 10
batch_size : 16
gpu : 1
learning rate : 0.001
checkpoint : 
max_words_cnt : 256
Allocating memory  
for tensors : 36609236 bytes, 
for c_tensors: 3194706328 bytes 
for grad_tensors: 1241779004 bytes
epoch 0 :  [224/256]loss : 5.54111
epoch 1 :  [224/256]loss : 1.36544
epoch 2 :  [224/256]loss : 0.178868
epoch 3 :  [224/256]loss : 0.0472531
epoch 4 :  [224/256]loss : 0.0245251
epoch 5 :  [224/256]loss : 0.0195127
epoch 6 :  [224/256]loss : 0.0174135
epoch 7 :  [224/256]loss : 0.0162055
epoch 8 :  [224/256]loss : 0.0154597
epoch 9 :  [224/256]loss : 0.0147902
checkpoint saved : ./checkpoints/checkpoint_20250608_200259_9.bin
```

### inference
```
$ ./lm -e 0 -c ./checkpoints/checkpoint_20250608_200259_9.bin
corpus : ./resources/time_machine/timemachine_preprocessed.txt
epochs : 0
batch_size : 16
gpu : 1
learning rate : 0.001
checkpoint : ./checkpoints/checkpoint_20250608_200259_9.bin
max_words_cnt : 256
Allocating memory  
for tensors : 36355416 bytes, 
for c_tensors: 17206900 bytes 
for grad_tensors: 14209596 bytes
loading from checkpoint : ./checkpoints/checkpoint_20250608_200259_9.bin
loaded from checkpoint
serving mode
test file : ./test_lm.txt
sentence : the time machine
by h g wells i the time traveller for so it will be convenient to speak of him was expounding a recondite matter to us his grey eyes shone and twinkled and his usually pale face was flushed and animated the fire burned brightly and animated the fire burned brightly 
-----------------
```

### pre-trained lm model

* [model link1](https://cpp-transformer-1252366230.cos.ap-beijing.myqcloud.com/lm/checkpoint_20250617_162040_7.bin)
* [model link2](https://cpp-transformer-us-1252366230.cos.na-ashburn.myqcloud.com/lm/checkpoint_20250617_162040_7.bin)

This model was trained for 8 epochs using the full text of The Time Machine novel.

## handwritten_recognition

To verify some functions more quickly, I have introduced a handwritten digit recognition program.

```
./handwritten_recognition 
images magic : 2051
label magic : 2049
lables_num : 60000
data loaded.
Actions:
...
evaluating :  [10000/10000] correct : 9501
epoch : 9 [50000/50000] loss : 0.150985
evaluating :  [10000/10000] correct : 9493
```

### graphviz supported

You can add a line of code like this to the program to output an out.dot file that records the tensor computation topology. For example, in mnist.cpp:

```
printAllActions();
printDotGraph(); // here
allocMemAndInitTensors();
```

If you have Graphviz installed, you can use the following command to convert the out.dot file into a PNG image:

```
dot -Tpng out.dot -o out.png
```

Here's an example from my side where a PNG file is generated as output.

![alt text](handwritten_recognition_topo.png)

## legacy version

[v1](https://github.com/freelw/cpp-transformer/tree/v1_freeze_20250529)


## Derivation of backpropagation gradient formulas

* [Derivation](doc/equations/readme.md)

## Reference Materials

* [Dive into Deep Learning](https://d2l.ai/)
* [recognizing_handwritten_digits](https://github.com/freelw/recognizing_handwritten_digits)
* [micrograd](https://github.com/EurekaLabsAI/micrograd)
