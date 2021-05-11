# experience result

pretrain model: nezha-large 
pretrain data: enhance data with closure  align text from !
pretrain strategy: 2-gram

co-ocurrence stratagy: not in futher pretrain and we use it in fineturning 

kfold-0 check-point-30000 fgm 0.1  enhanced-data  NeZhaForSequenceClassification dropout 0.5 best model  线下 0。96 我觉得可以直接pass

kfold-1 check-point-30000 fgm 0.1  enhanced-data  NeZhaForSequenceClassification dropout 0.2 0.906249
                                                                                             best model  0.899228
                                                            
kfold-2 check-point-30000 fgm 0.1  enhanced-data  NeZhaForSequenceClassification dropout 0.2 mutil-sample dropout 0.907313
                                                                                             best model  

kfold-3 check-point-30000 fgm 0.1                     enhanced-data  NeZhaForSequenceClassification ids dropout 0.2 mutil-sample dropout 0.907465

kfold-4 check-point-30000 pgd epsilon=0.5, alpha=0.3  enhanced-data  NeZhaForSequenceClassification dropout 0.2 mutil-sample dropout 0.907439


kfold-5 check-point-30000 pgd epsilon=0.5, alpha=0.3  enhanced-data  cls cat(with pooler)  dropout 0.2 mutil-sample dropout randomseed = 2021 0.907217

kfold-6 check-point-30000 fgm 0.5                     enhanced-data  cls cat(with pooler)  dropout 0.2 mutil-sample dropout randomseed = 2021  0.908254

kfold-7 check-point-30000 fgm 0.2                     enhanced-data  cls cat(with pooler)  dropout 0.2 mutil-sample dropout randomseed = 42 0.907177

kfold-8 check-point-30000 fgm 0.2                     enhanced-data+chancy data  cls cat(with pooler)  dropout 0.2 mutil-sample dropout randomseed = 2021  0.905666

kfold-9 check-point-30000 fgm 0.1                     enhanced-data  ElectraForSequenceClassification 0.898179

kfold-10 check-point-30000 fgm 0.5                     enhanced-data  cls cat(with pooler)+co-ocurrence ids  dropout 0.2 mutil-sample dropout randomseed = 2021 0.908779

kfold-11 check-point-30000 fgm 0.2                     enhanced-data  cls cat(with pooler)+co-ocurrence ids  dropout 0.2 mutil-sample dropout randomseed = 2021 0.909076

kfold-12 check-point-30000 fgm 0.2                     enhanced-data+chancy data(neg)  cls cat(with pooler)  dropout 0.2 mutil-sample dropout randomseed = 2021 0.897528

kfold-13 check-point-30000 fgm 0.3                     enhanced-data  cls cat(with pooler)+co-ocurrence ids  dropout 0.2 mutil-sample dropout randomseed = 2021 0.907214

result-enhance 9+11 0.910

result-enhance-roberta-nezha-electra-zzdu 0.911729

kfold-14 roberta check-point-20000 fgm 0.3             enhanced-data  cls cat(with pooler)+co-ocurrence ids  dropout 0.2 mutil-sample dropout randomseed = 2021 0.902357

kfold-15 check-point-30000 pgd epsilon=0.2, alpha=0.3  enhanced-data  cls cat(with pooler)+co-ocurrence ids  dropout 0.2 mutil-sample dropout randomseed = 2021 0.907543

kfold-16 electra-point-30000 fgm 0.2                   enhanced-data  cls cat(with pooler)+co-ocurrence ids  dropout 0.2 mutil-sample dropout randomseed = 2021 

kfold-17 electra-point-30000 pgd epsilon=0.2,alpha=0.3 enhanced-data  cls cat(with pooler)+co-ocurrence ids  dropout 0.2 mutil-sample dropout randomseed = 2021  0。901

kfold-18 models/roberta      fgm 0.2                   enhanced-data  cls cat(with pooler)+co-ocurrence ids  dropout 0.2 mutil-sample dropout randomseed = 2021 

0.913046


15+14+16+11+zzdu 0.913697

pgd-nezha roberta electra fgm-nezha nezha

```text
PGD 参数多，batch_size 需要小一点，如果2080Ti 需要24， 2090需要48
FGM 参数不是特别多，一般为32或者64
```

normalize embedding