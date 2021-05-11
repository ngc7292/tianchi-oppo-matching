# experience result

pretrain model: nezha-large 
pretrain data: enhance data with closure  align text from !
pretrain strategy: 2-gram

co-ocurrence stratagy: not in futher pretrain and we use it in fineturning 

kfold-1 原始zzdu的模型与代码 fgm 0.5【0.908】
kfold-2 zzdu的代码加上co-ocurrence embedding-十折 
kfold-3 从！开始对应词典预训练checkpoint30000 loss0。3 + co-ocurrence 0.9088


