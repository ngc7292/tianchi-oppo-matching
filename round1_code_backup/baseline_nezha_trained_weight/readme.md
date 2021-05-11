# README

## 实验方法
mlm+fineturn
mlm采用数据增强，即将句子正反交换，得到250000条数据，根据这些数据进行预训练

预训练采用4卡3090，训练500个epoch，采用句子对训练，不采用n-gram mask，就是简单的直接伯努利采样进行mask
模型方面采用bert进行训练，不使用预置权重，随机初始化，

fineturn过程

--｜--｜--｜
id | 训练方法 ｜ decoder层 ｜ 备注
1 | 无 ｜ 两层mlp 一层将其扩展为2048 一层到2（MLP有用吗）| 不知道是预训练还是什么的原因，反正没用
2 | 5-fold ｜ 两层mlp 一层将其扩展为2048 一层到2（MLP有用吗）| 
3 | 无 ｜ 一层MLP | 没试过



##  实验记录

预训练loss趋势

{'loss': 3.3609, 'learning_rate': 4.989775051124745e-05, 'epoch': 1.02}
{'loss': 1.4469, 'learning_rate': 4.979550102249489e-05, 'epoch': 2.04}
{'loss': 1.2024, 'learning_rate': 4.9693251533742335e-05, 'epoch': 3.07}
{'loss': 1.061, 'learning_rate': 4.9591002044989774e-05, 'epoch': 4.09}
{'loss': 0.9797, 'learning_rate': 4.948875255623722e-05, 'epoch': 5.11}
{'loss': 0.9205, 'learning_rate': 4.938650306748466e-05, 'epoch': 6.13}
{'loss': 0.8578, 'learning_rate': 4.928425357873211e-05, 'epoch': 7.16}
{'loss': 0.8195, 'learning_rate': 4.918200408997955e-05, 'epoch': 8.18}
{'loss': 0.7843, 'learning_rate': 4.907975460122699e-05, 'epoch': 9.2}
{'loss': 0.7584, 'learning_rate': 4.897750511247444e-05, 'epoch': 10.22}
{'loss': 0.7361, 'learning_rate': 4.8875255623721885e-05, 'epoch': 11.25}
{'loss': 0.7046, 'learning_rate': 4.877300613496933e-05, 'epoch': 12.27}
{'loss': 0.6917, 'learning_rate': 4.867075664621677e-05, 'epoch': 13.29}
{'loss': 0.6566, 'learning_rate': 4.856850715746422e-05, 'epoch': 14.31}
{'loss': 0.6479, 'learning_rate': 4.846625766871166e-05, 'epoch': 15.34}
{'loss': 0.6266, 'learning_rate': 4.8364008179959104e-05, 'epoch': 16.36}
{'loss': 0.6105, 'learning_rate': 4.826175869120655e-05, 'epoch': 17.38}
{'loss': 0.5998, 'learning_rate': 4.815950920245399e-05, 'epoch': 18.4}     
{'loss': 0.5755, 'learning_rate': 4.8057259713701436e-05, 'epoch': 19.43}   
{'loss': 0.5684, 'learning_rate': 4.7955010224948876e-05, 'epoch': 20.45}

{'loss': 0.3851, 'learning_rate': 4.5807770961145195e-05, 'epoch': 41.92}
{'loss': 0.3795, 'learning_rate': 4.570552147239264e-05, 'epoch': 42.94}
{'loss': 0.3737, 'learning_rate': 4.560327198364008e-05, 'epoch': 43.97}
{'loss': 0.3664, 'learning_rate': 4.550102249488753e-05, 'epoch': 44.99}
{'loss': 0.3615, 'learning_rate': 4.539877300613497e-05, 'epoch': 46.01}
{'loss': 0.3529, 'learning_rate': 4.5296523517382414e-05, 'epoch': 47.03}
{'loss': 0.3472, 'learning_rate': 4.519427402862986e-05, 'epoch': 48.06}
{'loss': 0.3444, 'learning_rate': 4.5092024539877307e-05, 'epoch': 49.08}
{'loss': 0.3408, 'learning_rate': 4.4989775051124746e-05, 'epoch': 50.1}
{'loss': 0.3334, 'learning_rate': 4.488752556237219e-05, 'epoch': 51.12}
{'loss': 0.3286, 'learning_rate': 4.478527607361964e-05, 'epoch': 52.15}
{'loss': 0.3221, 'learning_rate': 4.468302658486708e-05, 'epoch': 53.17}
{'loss': 0.3163, 'learning_rate': 4.4580777096114525e-05, 'epoch': 54.19}
{'loss': 0.3099, 'learning_rate': 4.4478527607361964e-05, 'epoch': 55.21}
{'loss': 0.3087, 'learning_rate': 4.437627811860941e-05, 'epoch': 56.24}
{'loss': 0.3032, 'learning_rate': 4.427402862985685e-05, 'epoch': 57.26}
{'loss': 0.2997, 'learning_rate': 4.41717791411043e-05, 'epoch': 58.28}
{'loss': 0.2956, 'learning_rate': 4.4069529652351736e-05, 'epoch': 59.3}
{'loss': 0.2892, 'learning_rate': 4.396728016359918e-05, 'epoch': 60.33}                                                                                                                                    
{'loss': 0.2846, 'learning_rate': 4.386503067484663e-05, 'epoch': 61.35}

{'loss': 0.2165, 'learning_rate': 4.171779141104294e-05, 'epoch': 82.82}
{'loss': 0.2123, 'learning_rate': 4.161554192229039e-05, 'epoch': 83.84}
{'loss': 0.2104, 'learning_rate': 4.1513292433537835e-05, 'epoch': 84.87}
{'loss': 0.2035, 'learning_rate': 4.1411042944785274e-05, 'epoch': 85.89}
{'loss': 0.2051, 'learning_rate': 4.130879345603272e-05, 'epoch': 86.91}
{'loss': 0.2038, 'learning_rate': 4.120654396728017e-05, 'epoch': 87.93}
{'loss': 0.1987, 'learning_rate': 4.1104294478527614e-05, 'epoch': 88.96}
{'loss': 0.199, 'learning_rate': 4.100204498977505e-05, 'epoch': 89.98}
{'loss': 0.1933, 'learning_rate': 4.08997955010225e-05, 'epoch': 91.0}
{'loss': 0.1931, 'learning_rate': 4.079754601226994e-05, 'epoch': 92.02}
{'loss': 0.1913, 'learning_rate': 4.0695296523517386e-05, 'epoch': 93.05}
{'loss': 0.1899, 'learning_rate': 4.059304703476483e-05, 'epoch': 94.07}
{'loss': 0.1879, 'learning_rate': 4.049079754601227e-05, 'epoch': 95.09}
{'loss': 0.1846, 'learning_rate': 4.038854805725972e-05, 'epoch': 96.11}
{'loss': 0.1836, 'learning_rate': 4.028629856850716e-05, 'epoch': 97.14}
{'loss': 0.1817, 'learning_rate': 4.0184049079754604e-05, 'epoch': 98.16}
{'loss': 0.1782, 'learning_rate': 4.0081799591002043e-05, 'epoch': 99.18}
{'loss': 0.178, 'learning_rate': 3.997955010224949e-05, 'epoch': 100.2}
{'loss': 0.175, 'learning_rate': 3.987730061349693e-05, 'epoch': 101.23}                                                                                                                                    
{'loss': 0.1735, 'learning_rate': 3.9775051124744376e-05, 'epoch': 102.25}

{'loss': 0.1451, 'learning_rate': 3.7627811860940696e-05, 'epoch': 123.72}
{'loss': 0.1428, 'learning_rate': 3.752556237218814e-05, 'epoch': 124.74}
{'loss': 0.14, 'learning_rate': 3.742331288343559e-05, 'epoch': 125.77}
{'loss': 0.141, 'learning_rate': 3.732106339468303e-05, 'epoch': 126.79}
{'loss': 0.14, 'learning_rate': 3.7218813905930474e-05, 'epoch': 127.81}
{'loss': 0.1375, 'learning_rate': 3.711656441717792e-05, 'epoch': 128.83}
{'loss': 0.1383, 'learning_rate': 3.701431492842536e-05, 'epoch': 129.86}
{'loss': 0.136, 'learning_rate': 3.6912065439672807e-05, 'epoch': 130.88}
{'loss': 0.1366, 'learning_rate': 3.6809815950920246e-05, 'epoch': 131.9}
{'loss': 0.1349, 'learning_rate': 3.670756646216769e-05, 'epoch': 132.92}
{'loss': 0.134, 'learning_rate': 3.660531697341513e-05, 'epoch': 133.95}
{'loss': 0.1323, 'learning_rate': 3.650306748466258e-05, 'epoch': 134.97}
{'loss': 0.1304, 'learning_rate': 3.6400817995910025e-05, 'epoch': 135.99}
{'loss': 0.1307, 'learning_rate': 3.6298568507157465e-05, 'epoch': 137.01}
{'loss': 0.1308, 'learning_rate': 3.619631901840491e-05, 'epoch': 138.04}
{'loss': 0.1289, 'learning_rate': 3.609406952965235e-05, 'epoch': 139.06}
{'loss': 0.1275, 'learning_rate': 3.59918200408998e-05, 'epoch': 140.08}
{'loss': 0.1272, 'learning_rate': 3.5889570552147236e-05, 'epoch': 141.1}
{'loss': 0.127, 'learning_rate': 3.578732106339468e-05, 'epoch': 142.13}
{'loss': 0.1268, 'learning_rate': 3.568507157464213e-05, 'epoch': 143.15}

{'loss': 0.1121, 'learning_rate': 3.353783231083845e-05, 'epoch': 164.62}
{'loss': 0.1091, 'learning_rate': 3.3435582822085895e-05, 'epoch': 165.64}
{'loss': 0.1079, 'learning_rate': 3.3333333333333335e-05, 'epoch': 166.67}
{'loss': 0.1102, 'learning_rate': 3.323108384458078e-05, 'epoch': 167.69}
{'loss': 0.1073, 'learning_rate': 3.312883435582822e-05, 'epoch': 168.71}
{'loss': 0.1079, 'learning_rate': 3.302658486707567e-05, 'epoch': 169.73}
{'loss': 0.1043, 'learning_rate': 3.2924335378323114e-05, 'epoch': 170.76}
{'loss': 0.1061, 'learning_rate': 3.282208588957055e-05, 'epoch': 171.78}
{'loss': 0.1052, 'learning_rate': 3.2719836400818e-05, 'epoch': 172.8}
{'loss': 0.1029, 'learning_rate': 3.261758691206544e-05, 'epoch': 173.82}
{'loss': 0.1036, 'learning_rate': 3.2515337423312886e-05, 'epoch': 174.85}
{'loss': 0.1037, 'learning_rate': 3.2413087934560325e-05, 'epoch': 175.87}
{'loss': 0.1027, 'learning_rate': 3.231083844580777e-05, 'epoch': 176.89}
{'loss': 0.1023, 'learning_rate': 3.220858895705521e-05, 'epoch': 177.91}
{'loss': 0.104, 'learning_rate': 3.210633946830266e-05, 'epoch': 178.94}



## todo
-[] add NSP to pretrain the id data 
