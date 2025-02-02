#!/bin/bash

# python unsupervised_run.py zh_en N
# python unsupervised_run.py ja_en N
# python unsupervised_run.py fr_en N

#python unsupervised_run.py zh_en V
#python unsupervised_run.py ja_en V
#python unsupervised_run.py fr_en V

#python unsupervised_run.py zh_en V 0.6
#python unsupervised_run.py zh_en V 0.7
#python unsupervised_run.py zh_en V 0.8
#python unsupervised_run.py zh_en V 0.9
#
#python unsupervised_run.py ja_en V 0.6
#python unsupervised_run.py ja_en V 0.7
#python unsupervised_run.py ja_en V 0.8
#python unsupervised_run.py ja_en V 0.9
#
#python unsupervised_run.py fr_en V 0.6
#python unsupervised_run.py fr_en V 0.7
#python unsupervised_run.py fr_en V 0.8
#python unsupervised_run.py fr_en V 0.9


#python unsupervised_run.py zh_en V 0.2
#python unsupervised_run.py zh_en V 0.3
#python unsupervised_run.py zh_en V 0.4
#python unsupervised_run.py zh_en V 0.5
#
#python unsupervised_run.py ja_en V 0.2
#python unsupervised_run.py ja_en V 0.3
#python unsupervised_run.py ja_en V 0.4
#python unsupervised_run.py ja_en V 0.5

#python unsupervised_run.py fr_en V 0.2
#python unsupervised_run.py fr_en V 0.3
#python unsupervised_run.py fr_en V 0.4
#python unsupervised_run.py fr_en V 0.5
#
#python unsupervised_run.py zh_en V 0.1
#python unsupervised_run.py zh_en V 0.15
#
#python unsupervised_run.py ja_en V 0.1
#python unsupervised_run.py ja_en V 0.15
#
#python unsupervised_run.py fr_en V 0.1
#python unsupervised_run.py fr_en V 0.15

#python unsupervised_run.py zh_en V 0.2

# 2.5 1.8 1.3 2.5

#python unsupervised_run.py zh_en V 0.2 0.2 2.5 1.8 1.3 2.5
#python unsupervised_run.py zh_en V 0.2 0.3 2.5 1.8 1.3 2.5
#python unsupervised_run.py zh_en V 0.2 0.4 2.5 1.8 1.3 2.5
#python unsupervised_run.py zh_en V 0.2 0.5 2.5 1.8 1.3 2.5
#python unsupervised_run.py zh_en V 0.2 0.6 2.5 1.8 1.3 2.5
#python unsupervised_run.py zh_en V 0.2 0.7 2.5 1.8 1.3 2.5

# 2.15 2.5 1.5 2
#python unsupervised_run.py fr_en V 0.3 0.2 2.15 2.5 1.5 2
#python unsupervised_run.py fr_en V 0.3 0.3 2.15 2.5 1.5 2
#python unsupervised_run.py fr_en V 0.3 0.4 2.15 2.5 1.5 2
#python unsupervised_run.py fr_en V 0.3 0.5 2.15 2.5 1.5 2
#python unsupervised_run.py fr_en V 0.3 0.6 2.15 2.5 1.5 2
#python unsupervised_run.py fr_en V 0.3 0.7 2.15 2.5 1.5 2

#zh-en 增加两个dropout参数
#python unsupervised_run.py zh_en V 0.2 0.8 2.5 1.8 1.3 2.5
#python unsupervised_run.py zh_en V 0.2 0.9 2.5 1.8 1.3 2.5

#补ja-en实验
#python unsupervised_run.py ja_en V 0.3 0.2 2.15 2.5 1.5 2
#python unsupervised_run.py ja_en V 0.3 0.3 2.15 2.5 1.5 2
#python unsupervised_run.py ja_en V 0.3 0.4 2.15 2.5 1.5 2
#python unsupervised_run.py ja_en V 0.3 0.5 2.15 2.5 1.5 2
#python unsupervised_run.py ja_en V 0.3 0.6 2.15 2.5 1.5 2
#python unsupervised_run.py ja_en V 0.3 0.7 2.15 2.5 1.5 2
#python unsupervised_run.py ja_en V 0.3 0.8 2.15 2.5 1.5 2
#python unsupervised_run.py ja_en V 0.3 0.9 2.15 2.5 1.5 2

#关于N的sig对比
#dropout = 0.3
#python unsupervised_run.py zh_en N 0.2 0.3 2.5 1.8 1.3 2.5
#python unsupervised_run.py zh_en N 0.3 0.3 2.5 1.8 1.3 2.5
#python unsupervised_run.py zh_en N 0.4 0.3 2.5 1.8 1.3 2.5
#python unsupervised_run.py zh_en N 0.5 0.3 2.5 1.8 1.3 2.5
#python unsupervised_run.py zh_en N 0.6 0.3 2.5 1.8 1.3 2.5
#python unsupervised_run.py zh_en N 0.7 0.3 2.5 1.8 1.3 2.5

#有监督的sig对比
#python run.py zh_en 0.2 1 2.5 1.8 1.3 2.5
#python run.py zh_en 0.3 1 2.5 1.8 1.3 2.5
#python run.py zh_en 0.4 1 2.5 1.8 1.3 2.5
#python run.py zh_en 0.5 1 2.5 1.8 1.3 2.5
#python run.py zh_en 0.6 1 2.5 1.8 1.3 2.5
#python run.py zh_en 0.7 1 2.5 1.8 1.3 2.5
#python run.py zh_en 0.8 1 2.5 1.8 1.3 2.5
#python run.py zh_en 0.9 1 2.5 1.8 1.3 2.5
#
#
#python run.py ja_en 0.2 1 2.15 2.5 1.5 2
#python run.py ja_en 0.3 1 2.15 2.5 1.5 2
#python run.py ja_en 0.4 1 2.15 2.5 1.5 2
#python run.py ja_en 0.5 1 2.15 2.5 1.5 2
#python run.py ja_en 0.6 1 2.15 2.5 1.5 2
#python run.py ja_en 0.7 1 2.15 2.5 1.5 2
#python run.py ja_en 0.8 1 2.15 2.5 1.5 2
#python run.py ja_en 0.9 1 2.15 2.5 1.5 2


#python run.py fr_en 0.2 1 2.45 2.25 1.5 1
#python run.py fr_en 0.3 1 2.45 2.25 1.5 1
#python run.py fr_en 0.4 1 2.45 2.25 1.5 1
#python run.py fr_en 0.5 1 2.45 2.25 1.5 1
#python run.py fr_en 0.6 1 2.45 2.25 1.5 1
#python run.py fr_en 0.7 1 2.45 2.25 1.5 1
#python run.py fr_en 0.8 1 2.45 2.25 1.5 1
#python run.py fr_en 0.9 1 2.45 2.25 1.5 1

#python run.py zh_en 0.2 1 2.5 1.8 1.3 2.5
#python run.py ja_en 0.2 1 2.15 2.5 1.5 2
#python run.py fr_en 0.2 1 2.45 2.25 1.5 1

#python run.py zh_en 0.2 1 1 1 1 1
#python run.py ja_en 0.2 1 1 1 1 1
#python run.py fr_en 0.2 1 1 1 1 1


#python unsupervised_run.py zh_en V 0.2 0.7 2.5 1.8 1.3 2.5
#python unsupervised_run.py zh_en V 0.2 0.7 1 1 1 1

# seed预测方法_1
#python unsupervised_run.py zh_en V 0.2 0.7 2.5 1.8 1.3 2.5

#python unsupervised_run_1.py zh_en V 0.7 0.3 2.5 1.8 1.3 2.5
#python unsupervised_run_1.py zh_en N 0.7 0.3 2.5 1.8 1.3 2.5

python unsupervised_run_1.py zh_en V 0.2 0.3 1 1 1 1 1
python unsupervised_run_1.py zh_en V 0.3 0.3 1 1 1 1 1
python unsupervised_run_1.py zh_en V 0.4 0.3 1 1 1 1 1
python unsupervised_run_1.py zh_en V 0.5 0.3 1 1 1 1 1
python unsupervised_run_1.py zh_en V 0.6 0.3 1 1 1 1 1
python unsupervised_run_1.py zh_en V 0.7 0.3 1 1 1 1 1
python unsupervised_run_1.py zh_en V 0.8 0.3 1 1 1 1 1
python unsupervised_run_1.py zh_en V 0.9 0.3 1 1 1 1 1
#
python unsupervised_run_1.py zh_en N 0.2 0.3 1 1 1 1 1
python unsupervised_run_1.py zh_en N 0.3 0.3 1 1 1 1 1
python unsupervised_run_1.py zh_en N 0.4 0.3 1 1 1 1 1
python unsupervised_run_1.py zh_en N 0.5 0.3 1 1 1 1 1
python unsupervised_run_1.py zh_en N 0.6 0.3 1 1 1 1 1
python unsupervised_run_1.py zh_en N 0.7 0.3 1 1 1 1 1
python unsupervised_run_1.py zh_en N 0.8 0.3 1 1 1 1 1
python unsupervised_run_1.py zh_en N 0.9 0.3 1 1 1 1 1



