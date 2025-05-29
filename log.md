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