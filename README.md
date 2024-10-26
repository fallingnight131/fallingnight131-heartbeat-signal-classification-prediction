# 心跳信号分析预测

## 数据描述

训练集：train.csv

一共包含10万条数据，每条数据分为三个字段，分别是id、heartbeat_signal和label 。其中id是数据的序号，heartbeat_signal是心跳信号，是一个列表，label是心跳信号的分类标签，可以取[0,1,2,3]中的一个。

测试集：testA.csv

一共包含10万条数据，每条数据分为两个字段，分别是id和heartbeat_signal。其中id是数据的序号，heartbeat_signal是心跳信号，是一个列表，和去掉最后一列的训练集一样。

| Field | Description |
| -------- | -------- |
| id | 为心跳信号分配的唯一标识 |
| heartbeat_signals    | 心跳信号序列  |
| label    | 心跳信号类别（0、1、2、3） |

## 数据下载

[数据下载连接](http://slkc06hf4.hn-bkt.clouddn.com/%E5%BF%83%E8%B7%B3%E4%BF%A1%E5%8F%B7%E5%88%86%E6%9E%90%E9%A2%84%E6%B5%8B%E6%95%B0%E6%8D%AE%E9%9B%86.zip)

如果被阻止了，可以试试在比赛平台下载

[比赛网站](https://tianchi.aliyun.com/competition/entrance/531883/information)

## 评测标准

选手需提交4种不同心跳信号预测的概率，选手提交结果与实际心跳类型结果进行对比，求预测的概率与真实值差值的绝对值（越小越好）。

具体计算公式如下：

针对某一个信号，若真实值为[*y1,y2,y3,y4*],模型预测概率值为[*a1,a2,a3,a4*],那么该模型的平均指标*abs-sum*为：

<p align="center">
  <img src="http://slkc06hf4.hn-bkt.clouddn.com/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-21%20185254.png" />
</p>

例如，心跳信号为1，会通过编码转成[0,1,0,0]，预测不同心跳信号概率为[0.1,0.7,0.1,0.1],那么这个预测结果的*abs-sum*为:

*abs-sum*=|0.1-0|+|0.7-1|-|0.1-0|+|0.1-0|=0.6

## 结果提交

提交前请确保预测结果的格式与**sample_submit.csv**中的格式一致，以及提交文件后缀名为csv。

形式如下：

|id|label_0|label_1|label_2|label_3|
|--|-------|-------|-------|-------|
|100000|0|0|0|0|
|100001|0|0|0|0|
|100002|0|0|0|0|
|100003|0|0|0|0|

# 运行方法

在根目录增加一个名为datas的文件夹，并将testA.csv和train.csv放入

安装requirements.txt文件所包含的包

先通过train.py训练模型

再通过predict.py预测心跳信号类型

