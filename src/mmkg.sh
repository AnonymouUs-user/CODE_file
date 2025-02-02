

#2024年12月12日
#python run_12_11.py FB_DB 0.2 0.5 1 1 0
#python run_12_11.py FB_DB 0.2 0.5 1 1 1
#python run_12_11.py FB_DB 0.2 0.5 1 1 0.5
#python run_12_11.py FB_DB 0.2 0.5 1 1 0.1
#python run_12_11.py FB_DB 0.2 0.5 1 1 0.05
#python run_12_11.py FB_DB 0.2 0.5 1 1 0.005
# vae=false
#python run_12_11.py FB_DB 0.2 0.5 1 1 0


#2024年12月23日
# 有其他模态
#python run_hyper.py FB_DB 0.2 0.5 0 1
#纯结构模态
#python run_hyper.py FB_DB 0.2 0.5 0 0
#aug_loss的损失占比小一些
# 有其他模态
#python run_hyper.py FB_DB 0.2 0.5 0 1
#纯结构模态
#python run_hyper.py FB_DB 0.2 0.5 0 0

#2024年12月24日
# 测试aug_loss的占比权重
# 有其他模态
#python run_hyper.py FB_DB 0.2 0.5 0 1 0.5
#python run_hyper.py FB_DB 0.2 0.5 0 1 0.1
#python run_hyper.py FB_DB 0.2 0.5 0 1 0.05
#纯结构模态
#python run_hyper.py FB_DB 0.2 0.5 0 0 0.5
#python run_hyper.py FB_DB 0.2 0.5 0 0 0.1
#python run_hyper.py FB_DB 0.2 0.5 0 0 0.05
#python run_hyper.py FB_DB 0.2 0.5 0 0 2
#python run_hyper.py FB_DB 0.2 0.5 0 0 5
#原文default为300
#python run_hyper.py FB_DB 0.2 0.5 0 0 300
#python run_hyper.py FB_DB 0.2 0.5 0 0 30
#cosine计算方式
#python run_hyper.py FB_DB 0.2 0.5 0 0 30
#python run_hyper.py FB_DB 0.2 0.5 0 0 2
#inner比cosine的效果好一点
#增加inner_loss，并修改loss求和方式
#不增加inner_loss, 仅修改loss求和方式
#增加左侧的inner_loss
#python run_hyper.py FB_DB 0.2 0.5 0 0 2
#python run_hyper.py FB_DB 0.2 0.5 0 0 3
#python run_hyper.py FB_DB 0.2 0.5 0 0 4
#python run_hyper.py FB_DB 0.2 0.5 0 0 5
#python run_hyper.py FB_DB 0.2 0.5 0 0 1
#python run_hyper.py FB_DB 0.2 0.5 0 0 0.9
#python run_hyper.py FB_DB 0.2 0.5 0 0 0.8
#python run_hyper.py FB_DB 0.2 0.5 0 0 0.7
#python run_hyper.py FB_DB 0.2 0.5 0 0 0.6
#python run_hyper.py FB_DB 0.2 0.5 0 0 0.
#hyper参与到决策中
#双曲分支也生成伪标签
#python run_hyper.py FB_DB 0.2 0.5 0 0 1
#多模态
#python run_hyper.py FB_DB 0.2 0.5 0 1 1
# 更改hyper emb的初始化方式
#python run_hyper.py FB_DB 0.2 0.5 0 0 1


#2024年12月25日
#增加hyper-rel的嵌入 wrong 尺寸对不起来

#增加hyper-rel的嵌入
#python run_hyper.py FB_DB 0.2 0.5 0 0 1
#python run_hyper.py FB_DB 0.2 0.5 0 0 0.9
#python run_hyper.py FB_DB 0.2 0.5 0 0 0.7
#python run_hyper.py FB_DB 0.2 0.5 0 0 0.5
#python run_hyper.py FB_DB 0.2 0.5 0 0 0.3
#python run_hyper.py FB_DB 0.2 0.5 0 0 1.5
#python run_hyper.py FB_DB 0.2 0.5 0 0 2
#python run_hyper.py FB_DB 0.2 0.5 0 0 3
#python run_hyper.py FB_DB 0.2 0.5 0 0 4
#python run_hyper.py FB_DB 0.2 0.5 0 0 5

#2024年12月26日
#不同权重的调整参与到训练中weight_loss
#aug loss和align loss独立backward
#调整weight_loss的weight
#python run_hyper.py FB_DB 0.2 0.5 0 0 2
#python run_hyper.py FB_DB 0.2 0.5 0 0 5
#python run_hyper.py FB_DB 0.2 0.5 0 0 10
#python run_hyper.py FB_DB 0.2 0.5 0 0 20
#python run_hyper.py FB_DB 0.2 0.5 0 0 30
#python run_hyper.py FB_DB 0.2 0.5 0 0 40
#python run_hyper.py FB_DB 0.2 0.5 0 0 50
#python run_hyper.py FB_DB 0.2 0.5 0 0 60

#多模态
#python run_hyper.py FB_DB 0.2 0.5 0 1 20
#loss2单独backward augloss单独backward
#python run_hyper.py FB_DB 0.2 0.5 0 0 20
#python run_hyper.py FB_DB 0.2 0.5 0 1 20

#python run_0.2.py FB_DB 0.2 0.5 0 0


#2025年1月6日
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 5
#2025年1月7日
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 1
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 5
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 10
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 20
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 30

#最佳
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 20

#测试不同时间加入weight-loss，最后一项是epoch_weight
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 20 10

#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 20 20
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 20 25
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 20 30

#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 5 20
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 10 20
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 20 20
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 30 20

#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 35 20
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 40 20
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 50 20

#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 30 0
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 30 5
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 30 10
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 30 15
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 30 20
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 30 30

#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 30 0
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 30 5
#python run_weight_learn.py FB_DB 0.2 0.5 0.0 1 30 10
#
#python run_weight_learn.py FB_YAGO 0.2 0.5 0.0 1 30 0
#python run_weight_learn.py FB_YAGO 0.2 0.5 0.0 1 30 5
#python run_weight_learn.py FB_YAGO 0.2 0.5 0.0 1 30 10
#python run_weight_learn.py FB_YAGO 0.2 0.5 0.0 1 30 30

#python run_weight_learn.py FB_DB 0.5 0.5 0.0 1 30 5
#python run_weight_learn.py FB_DB 0.8 0.5 0.0 1 30 5
#python run_weight_learn.py FB_YAGO 0.5 0.5 0.0 1 30 5
#python run_weight_learn.py FB_YAGO 0.8 0.5 0.0 1 30 5

#csp=1 #csp=2 (24组) 0.2/0.5/0.8 两个编码器 1222开始
#python run_weight_learn.py FB_DB 0.5 Qwen7b 0.5 1 1 5 （1222）
#python run_weight_learn.py FB_DB 0.5 Qwen7b 0.5 2 1 5 (1232)
#python run_weight_learn.py FB_YAGO 0.5 Qwen7b 0.5 1 1 5 (1250)
#python run_weight_learn.py FB_YAGO 0.5 Qwen7b 0.5 2 1 5 (1303)

#python run_weight_learn.py FB_DB 0.8 Qwen7b 0.5 1 1 5 (1318)
#python run_weight_learn.py FB_DB 0.8 Qwen7b 0.5 2 1 5 (1334)
#python run_weight_learn.py FB_YAGO 0.8 Qwen7b 0.5 1 1 5 (1352)
#python run_weight_learn.py FB_YAGO 0.8 Qwen7b 0.5 2 1 5 (1407)

#python run_weight_learn.py FB_DB 0.2 Qwen7b 0.5 1 1 5 (1421)
#python run_weight_learn.py FB_DB 0.2 Qwen7b 0.5 2 1 5 (1437)
#python run_weight_learn.py FB_YAGO 0.2 Qwen7b 0.5 1 1 5 (1454)
#python run_weight_learn.py FB_YAGO 0.2 Qwen7b 0.5 2 1 5 (1508)

#python run_weight_learn.py FB_DB 0.2 llmembed 0.5 1 1 5
#python run_weight_learn.py FB_DB 0.2 llmembed 0.5 2 1 5
#python run_weight_learn.py FB_YAGO 0.2 llmembed 0.5 1 1 5
#python run_weight_learn.py FB_YAGO 0.2 llmembed 0.5 2 1 5

#python run_weight_learn.py FB_DB 0.5 llmembed 0.5 1 1 5 (1539)
#python run_weight_learn.py FB_DB 0.5 llmembed 0.5 2 1 5 (1555)
#python run_weight_learn.py FB_YAGO 0.5 llmembed 0.5 1 1 5 (1612)
#python run_weight_learn.py FB_YAGO 0.5 llmembed 0.5 2 1 5 (1626)

#python run_weight_learn.py FB_DB 0.8 llmembed 0.5 1 1 5
#python run_weight_learn.py FB_DB 0.8 llmembed 0.5 2 1 5
#python run_weight_learn.py FB_YAGO 0.8 llmembed 0.5 1 1 5
#python run_weight_learn.py FB_YAGO 0.8 llmembed 0.5 2 1 5

#2035
#2135
#python run_weight_learn.py FB_DB 0.2 Qwen7b 0.8 0 1 5
#python run_weight_learn.py FB_YAGO 0.2 Qwen7b 0.8 0 1 5

#csp1 csp2 2207
#python run_weight_learn.py FB_DB 0.2 Qwen7b 0.8 1 1 5
#python run_weight_learn.py FB_DB 0.2 Qwen7b 0.8 2 1 5
#
#python run_weight_learn.py FB_YAGO 0.2 Qwen7b 0.8 1 1 5
#python run_weight_learn.py FB_YAGO 0.2 Qwen7b 0.8 2 1 5

#Image的重做 2316
#1116
#python run_weight_learn.py FB_DB 0.2 llmembed 0.8 0 1 5
#python run_weight_learn.py FB_DB 0.2 llmembed 0.8 1 1 5
#python run_weight_learn.py FB_DB 0.2 llmembed 0.8 2 1 5
#python run_weight_learn.py FB_YAGO 0.2 llmembed 0.8 0 1 5
#python run_weight_learn.py FB_YAGO 0.2 llmembed 0.8 1 1 5
#python run_weight_learn.py FB_YAGO 0.2 llmembed 0.8 2 1 5

#python run_weight_learn.py FB_DB 0.2 llmembed 0.8 0 1 5
#python run_weight_learn.py FB_YAGO 0.2 llmembed 0.8 0 1 5
#python run_weight_learn.py FB_YAGO 0.8 Qwen7b 0.8 0 1 5

#w/o Name +csp1/csp2 1528开始
python run_weight_learn.py FB_DB 0.2 Qwen7b 0.8 1 1 5
python run_weight_learn.py FB_DB 0.2 Qwen7b 0.8 2 1 5
python run_weight_learn.py FB_DB 0.5 Qwen7b 0.8 1 1 5
python run_weight_learn.py FB_DB 0.5 Qwen7b 0.8 2 1 5
python run_weight_learn.py FB_DB 0.8 Qwen7b 0.8 1 1 5
python run_weight_learn.py FB_DB 0.8 Qwen7b 0.8 2 1 5

python run_weight_learn.py FB_YAGO 0.2 Qwen7b 0.8 1 1 5
python run_weight_learn.py FB_YAGO 0.2 Qwen7b 0.8 2 1 5
python run_weight_learn.py FB_YAGO 0.5 Qwen7b 0.8 1 1 5
python run_weight_learn.py FB_YAGO 0.5 Qwen7b 0.8 2 1 5
python run_weight_learn.py FB_YAGO 0.8 Qwen7b 0.8 1 1 5
python run_weight_learn.py FB_YAGO 0.8 Qwen7b 0.8 2 1 5

python run_weight_learn.py FB_DB 0.2 llmembed 0.8 1 1 5
python run_weight_learn.py FB_DB 0.2 llmembed 0.8 2 1 5
python run_weight_learn.py FB_DB 0.5 llmembed 0.8 1 1 5
python run_weight_learn.py FB_DB 0.5 llmembed 0.8 2 1 5
python run_weight_learn.py FB_DB 0.8 llmembed 0.8 1 1 5
python run_weight_learn.py FB_DB 0.8 llmembed 0.8 2 1 5

python run_weight_learn.py FB_YAGO 0.2 llmembed 0.8 1 1 5
python run_weight_learn.py FB_YAGO 0.2 llmembed 0.8 2 1 5
python run_weight_learn.py FB_YAGO 0.5 llmembed 0.8 1 1 5
python run_weight_learn.py FB_YAGO 0.5 llmembed 0.8 2 1 5
python run_weight_learn.py FB_YAGO 0.8 llmembed 0.8 1 1 5
python run_weight_learn.py FB_YAGO 0.8 llmembed 0.8 2 1 5