# experience result

pretrain model: nezha-large 
pretrain data: enhance data with closure  align text from !
pretrain strategy: 2-gram

co-ocurrence stratagy: not in futher pretrain and we use it in fineturning 

kfold check-point-30000 pgd embedding enhanced-data 5-fold 0.9062
kfold-2 check-point-40000 pgd embedding enhanced-data 5-fold 0.906104
kfold-3 check-point-40000 pgd embedding enhanced-data 10-fold 

kfold-4 check-point-30000 fgm 0.1 embedding enhanced-data 5-fold 0.905 ???
kfold-5 check-point-30000 pgd epsilon=0.1, alpha=0.3 embedding enhanced-data 5-fold 0.905843

kfold-6 check-point-30000 pgd epsilon=0.5, alpha=0.3 embedding enhanced-data 5-fold  
kfold-7 check-point-30000 fgm 0.5                              enhanced-data 5-fold  

kfold-8 check-point-30000 fgm 0.5            cat cls embedding enhanced-data 5-fold  
kfold-9 check-point-30000 fgm 0.5            cat add embedding enhanced-data 5-fold  

这个PGD怎么比FGM好啊，怪哦

