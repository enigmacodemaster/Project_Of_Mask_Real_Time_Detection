## week4作业-文字部分
> 20200807 欧阳紫洲

### 一、F.xx和nn.xx在使用时的区别

1. nn.CrossEntropyLoss()和F.cross_entropy()

使用方式：
```
import torch.nn as nn
import torch.nn.funtional as F

input_ = torch.randn(3,5, requires_grad=True)
target = torch.randint(5, (3,), dtype=torch.int64)

criterion = nn.CrossEntropyLoss() # 实例化
loss = criterion(input_, target)

loss = F.cross_entropy(input_,target)  # 直接调用函数

# 计算结果也相同
```

2. nn.NLLLoss()和F.nll_loss()的使用
```
nll = nn.NLLLoss()
input1 = torch.tensor([[-0.4076, -1.4076, -2.4076]])
target = torch.tensor([0])

res = nll(input1, target)

res = F.nll_loss(input1, target)

# 结果都为tensor(0.4076)

```

3. nn.Softmax和F.softmax
```
softFunc = nn.Softmax(dim=1)
input1 = torch.tensor([[3,5,7]]).float()

res1 = softFunc(input1)

res2 = F.softmax(input1, dim=1) # 按照行

# 结果相同

```


4. nn.LogSoftmax和F.log_softmax
```
input1 = torch.randn(2,3)
output = F.log_softmax(input1, dim=1)

log_sf = nn.LogSoftmax(dim = 1)
output = log_sf(input1)

# 二者的结果相同
```

### 二、F和nn的总结

#### 1. 从源码的层面看

nn中定义的是类，F中定义的是函数。二者同时存在的意义在于增加使用时的灵活性。定义网络层时，如果层内有需要维护的Variable时，用nn定义；反之，可以用nn.functional定义。nn.functional中的都是没有副作用无状态的函数，也就是说function内部一定是没有Variable的，而nn中不好说，一般都是nn.Module的子类，可以借助父类Module的方法方便的管理各种需要的状态变量。

使用nn的网络层时，无需自己维护和管理weight，但是使用F的相关层功能函数时，需要自己传入weight并做好维护。

比如我们看到，torch.nn.Conv2d的源码中，forward方法就是用F中的conv2d函数定义的。

```Python
import torch.nn.functional as F
class Conv2d(ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        # 细节不表
        pass
    
    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
```

#### 2. torch.sigmoid、torch.nn.sigmoid、torch.nn.functional.sigmoid的区别

三者从计算结果来看，没有区别，都是输入张量，返回tensor。使用nn.Sigmoid代表神经网络的一层，F.sigmoid和torch.sigmoid都是函数，而且F.sigmoid以后要弃用了，建议用torch.sigmoid。使用方式上，区别不大，就是nn.Sigmoid需要初始化一个类，F.sigmoid不需要。

#### 3. nn.CrossEntropyLoss()和F.cross_entropy()的区别

**参数：**
```
# F中的函数原型
torch.nn.functional.cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
```

* input: shape[N, C], N代表样本个数，C代表类别数目
* target: shape要求为大小为n的1-D tensor，包含类别的索引，要求 0 <= target[i] <= C-1;值得一提的是，**target元素必须是long类型的**。
* weight: 分别制定每个类别占的loss权重。1-D tensor。n个元素，代表n个类别的权重。如果训练样本不是很均衡的话，这个参数很有用。
* reduction='mean'，代表对N个样本的loss进行求平均之后返回；reduction='sum'，代表对N个样本的loss求和之后返回；'none'，代表直接返回n分样本的loss。

```
# nn中的原型
# 这是一个类
torch.nn.CrossEntropyLoss(weight:Optional[torch.Tensor] = None, size_average = None, ignore_index: int = -100, reduce = None, reduction: str = 'mean')
```

* 输入input还是模型的输出，包括每个类的得分，2-D Tensor，shape为[batch, N类]
* target还是大小为n的1D Tensor，包含的是类别的索引0到n-1。target也是类别值，不是one-hot编码。

**数学计算方法**：

```math
loss(x, class) = -\log(\frac{e^{x[class]}}{\displaystyle \sum_{j}{e^{x[j]}}})
```
从数学计算上来讲，torch中的交叉熵计算是logSoftmax和NllLoss的整合，等同于先计算log-Softmax，然后再将上一步的计算结果计算NLLLoss。


**作用和目的层面：**
判断实际输出（概率）和期望输出（概率）之间的接近程度（距离）。


#### 4. nn.NLLLoss()和F.nll_loss()的区别

**参数方面：** 一个是类，一个是函数
```
# nn
torch.nn.functional.nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
```
target: 类别索引，目标标签

input: 对数概率向量

```
# F
torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='elementwise_mean')
```
weight(Tensor): 为每个类别的loss设置权值，常用于类别不平衡的问题。weight必须是float类型的tensor。其长度要与类别C一致。

**作用和目的：** 
二者的作用是相同的。

把输入与label对应的那个值拿出来，去掉负号，求均值，或者求和Softmax计算出来的值范围在[0,1]，值的含义表示对应类别的概率，也就是说，每行（代表每张图片）中最接近于1的值对应的类别，就是该图片的概率最大的类别。经过log求取绝对值之后，就是最接近于0的值。如果此时每行中的最小值对应的类别值和label中的类别值相同，那么每行中的最小值求和取平均值最小，极端情况下就是0。input的预测值与target的值越接近，NLLLoss求出来的值就越接近0。

**数学公式**：

```math
l_n = -W_{n}x_{n,y_n}

W_n: 类别n的权重 

x_n: 目标类所对应的x中的值

y_n: 代表类别数量
```


#### 5. F.softmax和nn.Softmax

**传入参数：** 

F.softmax(input, dim=None, _stacklevel=3, dtype=None)

* input：张量
* dim: 1代表按行计算，0代表按列计算

nn.Softmax
* dim = 0，对一列所有元素进行softmax计算，并使得每一列所有元素的和为1
* dim = 1, 对一行所有元素进行softmax计算，并使得每一行所有元素的和为1
* 和F中的softmax的区别在于，nn中的dim是在类的实例化的时候传入的，而在F中是作为参数直接传入的。这一点在F和nn中的很多地方都能体现。

**作用目的：**

对n维输入张量运用softmax函数，将张量的每个元素缩放到（0,1）区间，且和为1，分为按行和按列计算。

**数学公式：**

```math
softmax(x_i) = \frac{exp(x_i)}{\sum_{j}{exp(x_j)}}
```


#### 6. nn.LogSoftmax和F.log_softmax


**作用目的：** 其实就是对softmax求出来的结果，再求一次log值。

**数学公式：**

```math
LogSoftMax(x_i) = \log(\frac{\exp(x_i)}{\sum_{j}{exp(x_j)}})
```

**参数和softmax参数类似，不再赘述。**


> 还有其他很多损失函数比如l1, l2损失，本质上，背后的数学原理基本一致，计算结果相同，但是使用方式略不同。