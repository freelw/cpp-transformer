# cpu 推理效果

```
(base) cs@cs-desktop:~/project/cpp-transformer$ time ./transformer -e 0 -c ./checkpoints/checkpoint_20250529_173547_2.bin
corpus : ./resources/fra_preprocessed.txt
epochs : 0
batch_size : 128
gpu : 1
learning rate : 0.001
checkpoint : ./checkpoints/checkpoint_20250529_173547_2.bin
enc_vocab_size : 7939
dec_vocab_size : 13387
bos_id : 3
eos_id : 1
src_pad_id : 0
tgt_pad_id : 0
predicting : true
batch_size : 1
Warning: GPU backend is not available in CPU build. Now use cpu instead!!!
loading from checkpoint : ./checkpoints/checkpoint_20250529_173547_2.bin
loaded from checkpoint
serving mode
test file : ./test.txt
go now . -> attendez maintenant maintenant jusqu'à maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant 
i know that it is highly unlikely that you'd ever want to go out -> je sais que c'est difficile . 
good job -> <unk> ces trucs sont neufs . 
how nice ! -> comment <unk> ! 

real    2m45.053s
user    2m44.820s
sys     0m0.196s
```

# gpu 推理效果

```
(base) cs@cs-desktop:~/project/cpp-transformer$ time ./transformer -e 0 -c ./checkpoints/checkpoint_20250529_173547_2.bin
corpus : ./resources/fra_preprocessed.txt
epochs : 0
batch_size : 128
gpu : 1
learning rate : 0.001
checkpoint : ./checkpoints/checkpoint_20250529_173547_2.bin
enc_vocab_size : 7939
dec_vocab_size : 13387
bos_id : 3
eos_id : 1
src_pad_id : 0
tgt_pad_id : 0
predicting : true
batch_size : 1
loading from checkpoint : ./checkpoints/checkpoint_20250529_173547_2.bin
loaded from checkpoint
serving mode
test file : ./test.txt
go now . -> attendez maintenant maintenant jusqu'à maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant maintenant 
i know that it is highly unlikely that you'd ever want to go out -> je sais que c'est difficile . 
good job -> <unk> ces trucs sont neufs . 
how nice ! -> comment <unk> ! 

real    0m1.971s
user    0m1.529s
sys     0m0.408s
```

# mac 单核训练
```
time ./transformer -
e 30
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
Warning: GPU backend is not available in CPU build. Now use cpu instead!!!
Allocating memory
for tensors : 24512008 bytes,
for c_tensors: 352023828 bytes
for grad_tensors: 115093172 bytes
epoch 0 :  [512/512]loss : 4.6005
epoch 1 :  [512/512]loss : 3.4976
epoch 2 :  [512/512]loss : 2.95839
epoch 3 :  [512/512]loss : 2.59663
epoch 4 :  [512/512]loss : 2.25127
epoch 5 :  [512/512]loss : 1.98858
epoch 6 :  [512/512]loss : 1.79615
epoch 7 :  [512/512]loss : 1.63834
epoch 8 :  [512/512]loss : 1.49839
epoch 9 :  [512/512]loss : 1.35638
epoch 10 :  [512/512]loss : 1.22534
epoch 11 :  [512/512]loss : 1.14479
epoch 12 :  [512/512]loss : 1.02828
epoch 13 :  [512/512]loss : 0.971333
epoch 14 :  [512/512]loss : 0.874484
epoch 15 :  [512/512]loss : 0.815368
epoch 16 :  [512/512]loss : 0.776708
epoch 17 :  [512/512]loss : 0.713817
epoch 18 :  [512/512]loss : 0.647416
epoch 19 :  [512/512]loss : 0.622634
epoch 20 :  [512/512]loss : 0.561409
epoch 21 :  [512/512]loss : 0.544885
epoch 22 :  [512/512]loss : 0.506623
epoch 23 :  [512/512]loss : 0.507323
epoch 24 :  [512/512]loss : 0.490506
epoch 25 :  [512/512]loss : 0.469571
epoch 26 :  [512/512]loss : 0.435137
epoch 27 :  [512/512]loss : 0.424559
epoch 28 :  [512/512]loss : 0.416458
epoch 29 :  [512/512]loss : 0.398394
checkpoint saved : ./checkpoints/checkpoint_20250608_174959_29.bin
./transformer -e 30  20280.60s user 70.12s system 99% cpu 5:41:35.34 total
```