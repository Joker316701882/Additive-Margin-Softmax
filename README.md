# Additive-Margin-Softmax
This is the implementation of paper &lt;Additive Margin Softmax for Face Verification>

Training logic is not provide here, it is highly inspired by Sandberg's [Facenet](https://github.com/davidsandberg/facenet), check it if you are interested. But pay attention to ![this](https://github.com/Joker316701882/Additive-Margin-Softmax/issues/1) when you read and use this implementation together with sandberg's code. 

Instead, 
**model structure** can be found at **resface.py** 
and 
**loss head** can be found at **AM-softmax.py**

## lfw accuracy

### 2018-2-11
Currently it only reaches **97.6%**. There might be some bugs, or some irregular preprocessings, when it reaches > 99%, detail configuration will be posted here.

### 2018-2-14
Now acc on lfw reaches **99.3%** with only use resface36 and flipped-concatenate validation.

### 2018-2-15
After fixing bugs in training code, finally resface20 can reach **99.33%** which only took 4 hours to converge.  
**Notice**:
This model is trained on vggface2 without removing overlaps between vggface2 and lfw, so the performance is little higher than reported in orginal paper **98.98%**(*m=0.35*) which trained on casia whose overlaps with lfw are removed.
![lfw](./tfboard/lfw_acc.png)

### 2018-2-17
Using **L-Resnet50E-IR** which was proposed in ![this paper](https://arxiv.org/abs/1801.07698) can reach **99.42%**. Also I noticed that alignment method is crucial to accuracy. The quality of alignment algorithm might be the bottleneck of modern face recognition system.

### 2018-2-28
Just for fun, I tried m=0.2 with Resface20, acc on lfw reaches **99.47%**. All experiments that I've done used **AdamOptimizer without weight decay**, *SGD(with/without momentum) or RMSProp* actually performed really bad in my experiments. My assumption is the difference of implementation of optimizer in different framework (e.g. caffe and tf).
