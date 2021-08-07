#### 一、基于 PaddleRec 框架的 DLRM 推荐算法复现

##### 1. 快速执行
```
cd PaddleRec
python -u tools/trainer.py -m models/rank/dlrm/config.yaml
```

![快速执行](https://tva1.sinaimg.cn/large/008i3skNly1gt89fsdiuvg312z0qggz3.gif)

##### 2. Sample Data 结果
```
# train step
2021-08-07 16:25:16,833 - INFO - epoch: 0 done, auc: 0.687211,accuracy: 0.000000, epoch time: 408.50 s
2021-08-07 16:25:19,996 - INFO - Already save model in output_model_dlrm/0

2021-08-07 16:32:08,402 - INFO - epoch: 1 done, auc: 0.769624,accuracy: 0.000000, epoch time: 408.41 s
2021-08-07 16:32:12,014 - INFO - Already save model in output_model_dlrm/1

2021-08-07 15:51:03,326 - INFO - epoch: 2 done, auc: 0.838696,accuracy: 0.000000, epoch time: 410.22 s
2021-08-07 15:51:05,438 - INFO - Already save model in output_model_dlrm/2

# infer step
2021-08-07 17:06:05,010 - INFO - epoch: 0 done, auc: 0.761627,accuracy: 0.000000, epoch time: 57.38 s
2021-08-07 17:07:02,043 - INFO - epoch: 1 done, auc: 0.791379,accuracy: 0.000000, epoch time: 57.03 s
2021-08-07 17:07:59,870 - INFO - epoch: 2 done, auc: 0.817173,accuracy: 0.000000, epoch time: 57.83 s
```
![训练过程](https://tva1.sinaimg.cn/large/008i3skNly1gt89kyvq3lg31360qc7wh.gif)


#### 二、DLRM 算法原理

待补充...

#### 三、复现记录
1. 2021-08-04 tensorflow2 版本 
- pytorch 实现代码翻译为 tensorflow2 版本， 在 criteo 测试集上取得 accuracy >= 0.80 & auc >= 0.79

2. 2021-08-06 基于 PaddleRec 框架版本
- 参考 tensorflow2 实现代码，基于 PaddleRec 框架实现 DLRM，在 sample data 上成功运行

3. 2021-08-07 paddle 版本跑 criteo 数据集 & ~~调参~~ 炼丹！
- batch_size: 128
- Ai-Studio CPU 跑的太慢了... 申请一下 GPU 资源



#### 四、参考资料
1. [PaddleRec 文档](README_CN.md)
2. [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/pdf/1906.00091v1.pdf)
3. [Criteo 数据集](https://github.com/PaddlePaddle/PaddleRec/blob/release/2.1.0/datasets/criteo/run.sh)
4. [DLRM Pytorch 实现](https://github.com/facebookresearch/dlrm)

