#### 作业说明
> by 欧阳紫洲

由于电脑显存较小，只训练了不到1个epoch，batch_size设置为1，训练了大概1千来张图片，loss较刚开始训练一直在下降。

###### 问题：
华为云平台OBS中的存储的coco数据集的数据，好像在notebook的terminal中无法完全sync进去，采用了代码同步的方式也不行，也就是如下方式也不行：
```
import moxing as mox
mox.file.copy_parallel("obs://bucket_name/dir", "/home/ma-user/work/dir")
```
提示是空间不够，not enough disk space.

请问有解决方案吗？谢谢。
