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

## quick start

### build

```
./build_all.sh 
```

### Perform inference using a pre-trained model

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

### Train with a small dataset

```
./train_tiny.sh
```

#### output

```
```

## Reference Materials
[Dive into Deep Learning](https://d2l.ai/)
[recognizing_handwritten_digits](https://github.com/freelw/recognizing_handwritten_digits)
