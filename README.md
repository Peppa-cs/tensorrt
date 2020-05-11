# tensorrt

训练模型：vgg16bn，具体代码见main.py

训练框架：pytorch
版本：1.1 1.2 - ONNX IR version 0.0.4
（版本说明）：Tensorrt需要的ONNX IR version 0.0.3对应的pytorch版本为1.0.0，由于pytorch1.0.0在转换为onnx模型时不支持flatten算子，而目前的Tensorrt版本只支持使用flatten算子而不能使用 out = out.view(out.size(0),-1)，因此会报出版本不对应的警告，目前还没有产生错误，具体代码见pytorch2onnx.py。Xavier Form 发布了Tensorrt7的Jetson版本，但是现在还是preview，等正式版发布应该可以使用view等算子。

部署框架：Tensorrt6

部署流程：
1. 在pytorch训练好模型后，使用pytorch自带的torch.onnx.export()函数得到onnx模型。

2. 使用Tensorrt onnxparser解析onnx模型并进行优化得到engine，可以设置运算精度FP32、FP16、INT8，其中GPU支持FP32、FP16、INT8，DLA仅支持FP16、INT8，DLA支持的最大batchsize是32。

3. pytorch需要的图片格式为CHW/RGB，在部署到Tensorrt后仍然是此格式，cifar10提供的bin数据集刚好是CHW/RGB不需要再进行额外的转换，只需要每个通道减去均值除以方差即可。

4. 根据batchsize做iteration = 10000 / batchsize 次迭代得到cifar10测试集上的正确率。
