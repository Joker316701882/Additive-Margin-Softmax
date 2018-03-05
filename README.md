# Additive-Margin-Softmax
This is the implementation of paper &lt;Additive Margin Softmax for Face Verification>

Training logic is highly inspired by Sandberg's [Facenet](https://github.com/davidsandberg/facenet), check it if you are interested.

**model structure** can be found at **./models/resface.py** 
and 
**loss head** can be found at **AM-softmax.py**

## Usage
### Step1: Align Dataset
See folder "align", this totally forked from [insightface](https://github.com/deepinsight/insightface/tree/master/src/align). The default image size is **(112,96)**, in this repository, all trained faces share same size **(112,96)**. Use align code to align your train data and validation data (like lfw) first. 
```
python align_lfw.py --input-dir [train data dir] --output-dir [aligned output dir]
```
You can use align_lfw.py to align both training set and lfw, don't worry about others lie align_insight, align_dlib.

### Step2: Train AM-softmax
Read **parse_arguments()** function carefully to confiure parameters. If you are new in face recognition, after aligning dataset, simply run this code, the default settings will help you solve the rest.
```
python train.py --data_dir [aligned train data] --random_clip --learning_rate -1 --learning_rate_schedule_file ./data/learning_rate_AM_softmax.txt --lfw_dir [aligned lfw data]
```
Also watch out that acc on lfw is not from cross validation. Read source code for more detail. Thanks Sandberg again for his extraordinary [code](https://github.com/davidsandberg/facenet).

## News
| Date     | Update |
|----------|--------|
| 2018-02-11 | Currently it only reaches **97.6%**. There might be some bugs, or some irregular preprocessings, when it reaches > 99%, detail configuration will be posted here. |
| 2018-02-14 | Now acc on lfw reaches **99.3%** with only use resface36 and flipped-concatenate validation. |
| 2018-02-15 | After fixing bugs in training code, finally resface20 can reach **99.33%** which only took 4 hours to converge. **Notice**:This model is trained on vggface2 without removing overlaps between vggface2 and lfw, so the performance is little higher than reported in orginal paper **98.98%**(*m=0.35*) which trained on casia whose overlaps with lfw are removed.|
| 2018-02-17 | Using **L-Resnet50E-IR** which was proposed in ![this paper](https://arxiv.org/abs/1801.07698) can reach **99.42%**. Also I noticed that alignment method is crucial to accuracy. The quality of alignment algorithm might be the bottleneck of modern face recognition system.|
| 2018&#8209;02&#8209;28 | Just for fun, I tried m=0.2 with Resface20, acc on lfw reaches **99.47%**. All experimens that I've done used **AdamOptimizer without weight decay**, *SGD(with/without momentum) or RMSProp* actually performed really bad in my experiments. My assumption is the difference of implementation of optimizer inside different frameworks (e.g. caffe and tf). |
| 2018-03-05 | Add trainin logic and align code.|
## lfw accuracy
![img](./tfboard/lfw_acc.png)
