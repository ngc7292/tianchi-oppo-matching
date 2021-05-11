# 代码说明

## 项目路径
.
├── code
│   ├── test # 预测执行文件夹
│   ├── train # 训练文件夹
│   ├── test.sh # 预测执行脚本
│   └── train.sh # 训练示例脚本 包含futher-pretrain与fineturning过程
├── prediction_result
├── tcdata
│   └── oppo_breeno_round1_data # 数据文件
└── user_data
    ├── model_data
    │   ├── chinese-electra-180g-large-discriminator # electra 原始模型权重
    │   ├── chinese-roberta-wwm-ext-large # roberta 原始模型权重
    │   ├── finturn_models # fineturn 后模型权重
    │   └── nezha-large-www # nezha原始模型权重
    └── tmp_data # 临时文件
        ├── raw_data # futher-pretrain 训练文件
        └── tokens # vocab文件

## 算法说明

### 预训练
预训练使用nezha-large-www，chinese-roberta-wwm-ext-large， chinese-electra-180g-large-discriminator三个模型进行预训练，在预训练阶段
使用n-gram mask进行mask处理，在预训练中保留mask language model一个任务，对以上三个模型进行训练。

NEZHA 大概训练30000step 左右 loss降至0.3 
ELECTRA 训练100epochs loss降至 0.2
ROBERTA 训练30000step loss降至0.3

### finetune

使用了闭包数据增强，即通过A-B=1，B-C=1 即认为A-C=1，对数据进行增强。

在模型方面，在BERT之后的embedding层添加了一个token-co-occurrence 标识符，对于前后句子都有的token，embedding过后的每一维度加1。

采用mutil-sample dropout dropout_num 取8

fineturn阶段微调了前面futher-pertrain过的三种模型，其中nezha通过fgm（epsilon=0.2）PGD（epsilon=0.5, alpha=0.3）进行对抗训练，ROBERTA以及ELECTRA通过fgm（epsilon=0.2）进行训练。

训练过程中，对数据进行5折分别训练(训练权重太大，由于提交大小限制，删除了后三个模型)

### predict

预测时对模型得到的结果简单加和平均。


## 系统依赖

操作系统 Linux 18ce69b9ecf4 4.4.0-142-generic #168-Ubuntu SMP Wed Jan 16 21:00:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux
PYTHON Python 3.7.0
CUDA Cuda compilation tools, release 11.1, V11.1.105
python依赖 详见requirement.txt