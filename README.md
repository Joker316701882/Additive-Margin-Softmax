# Additive-Margin-Softmax
This is the implementation of paper &lt;Additive Margin Softmax for Face Verification>

Training logic is not provide here, it is highly inspired by Sandberg's [Facenet](https://github.com/davidsandberg/facenet), check it if you are interested.

Instead, 
**model structure** can be found at **resface.py** 
and 
**loss head** can be found at **AM-softmax.py**

Currently it only reaches 97.6%. There might be some bugs, when it reaches > 99%, detail configuration will be posted here.
![lfw](./tfboard/lfw_acc.png)

