# 任务型对话机器人笔记

## TOPK TOPP
在自然语言生成（NLG）中，"topk"和"topp"通常用于指从给定集合中选择前k或前p项的技术。例如，top-k选择涉及从集合中选择具有最高分数或值的k个项目，而top-p选择涉及从具有发生概率最高的p个项目中选择。这些技术在NLG中常用于从大型数据集中选择最相关或重要的信息，并生成概括或传达该信息给用户的自然语言文本。

## 进行测试
python interact.py --no_cuda --model_path model/pytorch_model.bin  
python preprocess.py --train_path data/train1.txt --save_path data/train1.pkl  
//注意--val_num 必须大于等于一个batch_size的大小  
python train.py --epochs 40 --no_cuda --batch_size 8 --train_path data/train1.pkl --pretrained_model model/pytorch_model.bin --model_config model/config.json  
//测试一下
python interact.py --no_cuda --model_path model/epoch40/pytorch_model.bin --config_path model/epoch40/config.json



// 测试

```python
python -u interact.py --no_cuda --model_path ./model/quality-epoch-64/pytorch_model_quality_126.bin --config_path ./model/quality-epoch-64/config.json

```


```python
python interact.py --no_cuda --model_path ./model/quality-epoch-64/pytorch_model_quality_189.bin --config_path ./model/quality-epoch-64/config.json

```

```python
python interact.py --no_cuda --model_path ./model/quality-epoch-64/pytorch_model_quality_189.bin --config_path ./model/quality-epoch-64/config.json --topk 8 --max_len 255

```

# 测试 4W 模型


```python
python interact.py --no_cuda --model_path ./model/quality-epoch-64/pytorch_model_636262.bin --config_path ./model/quality-epoch-64/config.json --topk 8 --max_len 255 --repetition_penalty 2

```


当前能力：
- 可以回答保险相关的简单问题
- 不支持多轮对话
- 纠错能力
- 领域判断

可以做的工程优化


1、优化输入页面,提升响应速度[done]
    

2、添加问题的缓存能力[done]


3、提供一定程度的纠错


4、多轮对话能力微调


5、测试问题


6、添加常用的问候语

- 儿童买什么保险好

- 老人


买保险任务

1、意图：买 XX 类型的保险（非车险）
- U：我想买保险
- S：你想给谁买？
- U：给自己/给父母/给孩子/给宠物/房屋/汽车
- S：请问你的年龄是多少
- U：
2、办理赔任务

