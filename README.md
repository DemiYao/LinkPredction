# 基于Pair-wise的链接预测方法
&emsp;&emsp;这份Repository是我2021年秋季社会网络分析课程的期末大作业，为OGB榜单中链接预测的相关问题提供了相关的解决方法。
## Introduction
&emsp;&emsp;随着各种现实世界的应用，链接预测任务的重要性在过去十年中引起了研究界越来越多的关注。连接预测有多种应用场景，例如，链接预测方法可以帮助推断潜在的蛋白质-蛋白质交互从而有效地节省人工盲检带来的成本。 另外，链接预测技术可用于预测社交媒体用户之间的新关系，或在电子商务网站上发现潜在的用户到项目的关系，从而可以改进用户体验。

​        现有的神经链接预测方法非常注重设计更具表现力的神经网络架构，而问题的一些基本属性往往被忽视。 例如，大多数神经模型将链接预测视为二元分类问题，自然采用交叉熵损失函数。 然而，这种学习模式似乎不适合链接预测问题。 首先，由于大多数图的自然稀疏性，链接分类极其不平衡。虽然可以采用欠采样，但在采样过程中会有信息丢失以及采样的比例很难决定。 其次，大多数链路预测评估协议都做不旨在将正对标记为 1 而将负对标记为 0，而是要求对正对进行排名高于负对。 因此，使用交叉熵函数似乎不是那么直接链接预测任务的目标。

&emsp;&emsp;本次作业在OGB开源的数据集上复现模型PLNLP提出的解决链接预测任务的方法。

&emsp;&emsp;Open Graph Benchmark（简称 OGB）是斯坦福大学开源的Python库，包含了图机器学习的基准数据集、数据加载器和评估器，目的在于促进可扩展的、健壮的、可复现的研究。

&emsp;&emsp;OGB 包含了多种图机器学习的多种任务，并且涵盖从社会和信息网络到生物网络，分子图和知识图的各种领域。所有数据集都有特定的数据拆分和评估指标，从而提供统一的评估标准。

## File Description
### 数据集介绍
- ogbl-ddi

&emsp;&emsp;ogbl-ddi数据集是一个均匀,不加权的无向图,代表药物之间的相互作用网络。每个节点代表一个FDA-批准或实验药物。边代表药物之间的相互作用,可以解释为在一起服用两种药物的联合作用明显不同于药物独立行动的预期效果。
- ogbi-collab

&emsp;&emsp;ogbl-collab 数据集是一个无向图，代表MAG索引的作者之间协作网络的子集。 每个节点代表一个作者，边表示作者之间的合作。 所有节点都具有 128 维特征，通过对作者发表的论文的词嵌入进行平均获得。 所有的边都与两个元信息相关联：年份和边权重，代表当年发表的合着论文的数量。 该图可以看作是一个动态的多图，因为如果两个节点在一年以上的时间里协作，它们之间可能会有多条边。

### 数据集概览
|             | ogbl-ddi  | ogbl-collab |
| ----------- | :-------: | :---------: |
| Node Number |   4,267   |   235,868   |
| Link Number | 1,334,889 |  1,285,465  |
| Evaluator   |  Hit@20   |   Hit@50    |

## 相关工作
链接预测算法可以被总结为三种方法，分别是启发式规则算法、基于embedding算法和神经网络算法。
- **启发式规则算法**
大多数现有启发式方法的关键思想是基于节点的邻域信息衡量两个目标的相似性。 这些启发式方法的成功证明了目标节点对邻域信息的重要性。 但是，启发式方法通常由于其简单的形式，在处理不同类型网络时的适用性和表现力较弱，而且需要人为加入先验信息。 
- **基于embedding算法**
基于embedding的算法主要学习节点的隐式特征。这些方法通过从概率角度保留结构近邻来学习一般的潜在embedding，并通过将节点embedding组合为边的特征来进行链接的预测。 
- **基于神经网络算法**
近年来，随着神经网络的兴起，链接预测任务的神经网络算法大量涌现，提高了解决链接预测任务的准确度和泛化性。
## Methods
本作业参考的模型PLNLP分为四个组件，分别为邻域编码器、链路预测器、负采样和目标函数。
### 1 邻域编码器
邻域编码器可以选用邻居节点信息编码器（如GCN、Graphsage和GAT）和邻居节点对编码器（如SEAL、NIAN和HalpNet）。以上使用的邻域编码器都已经封装在代码中。
### 2 链接分数预测器
- 点积

 ![image](https://user-images.githubusercontent.com/62380677/147214099-d3167a3c-d670-4a19-a95f-17deea1a29b8.png)

- 双线性点积

 ![image](https://user-images.githubusercontent.com/62380677/147214131-a55db04d-6048-48e1-8346-46c7a66f0579.png)

- MLP

 ![image](https://user-images.githubusercontent.com/62380677/147214164-b426d007-cc68-4fb9-9e85-2750c39258e5.png)

### 3 负采样
在实际训练中，难以取得真正的负样本，传统的方法使用随机采样，随机抽取具有未知链接关系的节点对作为负样本。
- 全局负采样
- 局部负采样
- 对抗性负采样
- 共享负采样

### 4 基于Pair-wise的目标函数
由于网络的稀疏性，链路对和非链路对之间往往存在着极大的不平衡。同时，大多数链接预测任务的目标不是将正对标记为1，将负对标记为0，而是要求将正对的排名高于负对。为了与链路预测的总体目标保持一致，采用了模型学习的排名思想，其形式化如下：

![image](https://user-images.githubusercontent.com/62380677/147214369-9fbea229-969a-4cf3-9c74-4cfdcd8c0732.png)

事实上，上述学习目标相当于最大化曲线下面积（AUC），即为正样本排名高于负样本的概率。优化AUC并不简单，因为此函数的梯度为零或未定义。已经提出了各种技术来使用替代函数来近似AUC。代理函数有几种可能的选择，如成对铰链损失、逻辑损失或指数损失。在本实验中，简单地选择了平方最小代理损失，这在理论上与AUC是一致的，那么本实验选取的目标函数就是：

![image](https://user-images.githubusercontent.com/62380677/147214397-dfb76b04-8b92-4f2c-b69f-0a5aba23b4a6.png)

为了防止过拟合问题，在目标函数中加上权重为λ的参数$L_2$正则化。实验中使用随机梯度下降（SGD）根据目标函数对于模型参数$\theta$进行优化。

## 运行环境
The code is implemented with PyTorch and PyTorch Geometric. Requirments:  
&emsp;1. python=3.6  
&emsp;2. pytorch=1.7.1  
&emsp;3. ogb=1.3.2  
&emsp;4. pyg=2.0.1


## 运行口令
ogbl-ddi:  

    python main.py --data_name=ogbl-ddi --emb_hidden_channels=512 --gnn_hidden_channels=512 --mlp_hidden_channels=512 --num_neg=3 --epochs=500 --neg_sampler=global --dropout=0.3 

ogbl-collab: 

    python main.py --data_name=ogbl-collab --predictor=DOT --use_valedges_as_input=True --year=2010 --train_on_subgraph=True --num_neg=1 --epochs=800 --eval_last_best=True --neg_sampler=global --dropout=0.3

## 运行结果
### ddi
![image](https://user-images.githubusercontent.com/62380677/147214600-15cd810b-b107-4b32-8133-90b89eceb176.png)

### collab
![image](https://user-images.githubusercontent.com/62380677/147214697-2cf3b08c-d883-477e-a0f9-b0f1d80dd9cb.png)



|            | ogbl-ddi (Hits@20) | ogbl-collab (Hits@50) |
| ---------- | :----------------: | :-------------------: |
| Validation |    81.84 ± 1.73    |     99.99 ± 0.01      |
| Test       |    90.75 ± 5.02    |     67.93 ± 0.81      |

## Conclusion

在数据集ddi和数据集collab中，本实验所采取的基于pair-wise的链接预测方法可以取得很好的结果。

## Reference
[1] Wang Z, Zhou Y, Hong L, et al. Pairwise Learning for Neural Link Prediction[J]. arXiv preprint arXiv:2112.02936, 2021.

[2] Yang J H, Chen C M, Wang C J, et al. HOP-rec: high-order proximity for implicit recommendation[C]//Proceedings of the 12th ACM Conference on Recommender Systems. 2018: 140-144.

[3] Adamic L A, Adar E. Friends and neighbors on the web[J]. Social networks, 2003, 25(3): 211-230.

[4] Aditya Grover and Jure Leskovec. node2vec: Scalable feature learning for networks. InProceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining,
pages 855–864. ACM, 2016.

[5] Thomas N Kipf and Max Welling. V ariational graph auto-encoders.NIPS Workshop on Bayesian
Deep Learning, 2016b.