# Transformer

## bug 记录

1. linear 的bias不要全是0，要初始化成随机值打破平衡性，否则不好收敛
2. lr 从 0.005 调整到 0.001 效果很好

## train

初步观察小数据可以收敛

```
(d2l) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/transformer$ ./transformer -e 10 -f ../../resources/fra_tiny.txt 
OMP_THREADS: 8
epochs : 10
dropout : 0.2
lr : 0.005
tiny : 0
data loaded
parameter size = 21388
[300/300]checkpoint saved : ./checkpoints/checkpoint_20250331_204059_0.bin
epoch 0 loss : 8.38027 emit_clip : 3
[300/300]epoch 1 loss : 4.6412 emit_clip : 3
[300/300]epoch 2 loss : 3.9675 emit_clip : 3
[300/300]epoch 3 loss : 3.90752 emit_clip : 3
[300/300]epoch 4 loss : 3.8816 emit_clip : 3
[300/300]epoch 5 loss : 3.874 emit_clip : 3
[300/300]epoch 6 loss : 3.85435 emit_clip : 3
[300/300]epoch 7 loss : 3.82307 emit_clip : 3
[300/300]epoch 8 loss : 3.81192 emit_clip : 3
[300/300]checkpoint saved : ./checkpoints/checkpoint_20250331_204059_9.bin
```

## 切割模型文件

```
split -b $(($(stat -c%s ./checkpoint_20250402_150847_40.bin)/2)) checkpoint_20250402_150847_40.bin checkpoint_20250402_150847_40_part_
```

## 合并文件

```
cat checkpoint_20250402_150847_40_part_aa checkpoint_20250402_150847_40_part_ab > checkpoint_20250402_150847_40.bin
```
## predict

```
(base) cs@cs-desktop:~/project/recognizing_handwritten_digits/c++/transformer$ ./transformer -e 0 -c ./checkpoint_20250402_150847_40.bin
OMP_THREADS: 8
epochs : 0
dropout : 0.2
lr : 0.001
tiny : 0
data loaded
warmUp done
parameter size = 21388
all parameters require_grad = true
loading from checkpoint : ./checkpoint_20250402_150847_40.bin
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

## perf
![alt text](perf/p_3681797.svg)

## 用完整数据训练30轮

```
(base) cs@cs-desktop:~/project/cpp-transformer$ time ./transformer -e 30
OMP_THREADS: 8
epochs : 30
dropout : 0.2
lr : 0.001
tiny : 0
data loaded
warmUp done
parameter size = 21388
all parameters require_grad = true
[139392/167130]
[167130/167130]checkpoint saved : ./checkpoints/checkpoint_20250402_171839_0.bin
epoch 0 loss : 4.29504 emit_clip : 1173
[167130/167130]epoch 1 loss : 3.45483 emit_clip : 1181
[167130/167130]epoch 2 loss : 3.11291 emit_clip : 1278
[167130/167130]epoch 3 loss : 2.88984 emit_clip : 1289
[167130/167130]epoch 4 loss : 2.69937 emit_clip : 1296
[167130/167130]epoch 5 loss : 2.5308 emit_clip : 1297
[167130/167130]epoch 6 loss : 2.37853 emit_clip : 1297
[167130/167130]epoch 7 loss : 2.21523 emit_clip : 1302
[167130/167130]epoch 8 loss : 2.07645 emit_clip : 1300
[167130/167130]epoch 9 loss : 1.94702 emit_clip : 1298
[167130/167130]checkpoint saved : ./checkpoints/checkpoint_20250402_171839_10.bin
epoch 10 loss : 1.84002 emit_clip : 1297
[167130/167130]epoch 11 loss : 1.75559 emit_clip : 1294
[167130/167130]epoch 12 loss : 1.67983 emit_clip : 1296
[167130/167130]epoch 13 loss : 1.60608 emit_clip : 1291
[167130/167130]epoch 14 loss : 1.55091 emit_clip : 1285
[167130/167130]epoch 15 loss : 1.50655 emit_clip : 1288
[167130/167130]epoch 16 loss : 1.46157 emit_clip : 1278
[167130/167130]epoch 17 loss : 1.42421 emit_clip : 1285
[167130/167130]epoch 18 loss : 1.39032 emit_clip : 1279
[167130/167130]epoch 19 loss : 1.36256 emit_clip : 1276
[167130/167130]checkpoint saved : ./checkpoints/checkpoint_20250402_171839_20.bin
epoch 20 loss : 1.33565 emit_clip : 1277
[167130/167130]epoch 21 loss : 1.30881 emit_clip : 1285
[167130/167130]epoch 22 loss : 1.29369 emit_clip : 1275
[167130/167130]epoch 23 loss : 1.2749 emit_clip : 1283
[167130/167130]epoch 24 loss : 1.25101 emit_clip : 1276
[167130/167130]epoch 25 loss : 1.23262 emit_clip : 1277
[167130/167130]epoch 26 loss : 1.21892 emit_clip : 1271
[167130/167130]epoch 27 loss : 1.19993 emit_clip : 1278
[167130/167130]epoch 28 loss : 1.19129 emit_clip : 1281
[167130/167130]checkpoint saved : ./checkpoints/checkpoint_20250402_171839_29.bin
epoch 29 loss : 1.17262 emit_clip : 1275

real    9812m22.864s
user    57377m44.498s
sys     320m49.835s
```

## 切割完整模型文件

```
split -b $(($(stat -c%s ./checkpoints/checkpoint_20250402_171839_29.bin)/2)) ./checkpoints/checkpoint_20250402_171839_29.bin checkpoint_20250402_171839_29_part_
```

## 合并文件

```
cat checkpoint_20250402_171839_29_part_aa checkpoint_20250402_171839_29_part_ab > checkpoint_20250402_171839_29.bin
```

## 推理

```
(base) cs@cs-desktop:~/project/cpp-transformer$ ./transformer -e 0 -c ./checkpoint_20250402_171839_29.bin 
OMP_THREADS: 8
epochs : 0
dropout : 0.2
lr : 0.001
tiny : 0
data loaded
warmUp done
parameter size = 21388
all parameters require_grad = true
loading from checkpoint : ./checkpoint_20250402_171839_29.bin
loaded from checkpoint
serving mode
go now . <eos> 
translate res : <bos> maintenant va aller maintenant va aller maintenant va aller maintenant va maintenant va maintenant va maintenant va maintenant va maintenant 
i try . <eos> 
translate res : <bos> j'essaie de essayer de m'occuper de ce que j'ai essayé de ce que j'ai essayé de j'ai essayé de j'ai 
cheers ! <eos> 
translate res : <bos> veuillez arrêter , vous vous vous vous vous vous vous vous vous vous vous vous vous vous vous vous vous 
get up . <eos> 
translate res : <bos> lève-toi <unk> <unk> d'air conditionné ! détends-toi ! <eos> 
hug me . <eos> 
translate res : <bos> permettez-moi de vérifier des commentaires <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> 
i know . <eos> 
translate res : <bos> je sais que j'ai des aventures j'ai entendu parler . <eos> 
no way ! <eos> 
translate res : <bos> aucun sens l'humour ne veut et ta santé et l'eau ne parfum ne parfum ne parfum ne parfum ne parfum 
be nice . <eos> 
translate res : <bos> soyez chouette sera sympa soyez sympa soyez sympa soyez sympa soyez sympa soyez sympa soyez sympa soyez sympa soyez sympa 
i jumped . <eos> 
translate res : <bos> j'ai sauté sous j'ai sauté , j'ai commandé entendu entendre , hier , j'ai sauté du réveil , j'ai sauté 
congratulations ! <eos> 
translate res : <bos> félicitations félicitations et félicitations félicitations félicitations félicitations félicitations félicitations félicitations félicitations félicitations félicitations félicitations félicitations félicitations félicitations félicitations félicitations félicitations
```

