import tensorflow as tf

def AM_logits_compute(bottle_neck, label_batch, args, nrof_classes, lamb):
    
    '''
    loss head proposed in paper:<Additive Margin Softmax for Face Verification>
    link: https://arxiv.org/abs/1801.05599

    bottle_neck : bottle_neck of Facenet, can be output of resface model
    label_batch : ground truth label of current training batch
    args:         arguments from cmd line
    nrof_classes: number of classes
    lamb: lambda: for controlling weights of standard softmax and AM-softmax (notice: in Additvie margin softmax
                  paper, author mentioned that there is no need to use lambda, with directly training using AM-softmax 
                  can also work) 
    '''
    m = 0.25
    s = 30
    with tf.name_scope('AM_logits'):
        kernel = tf.Variable(tf.truncated_normal([args.embedding_size, nrof_classes]))
        kernel_len = tf.sqrt(tf.reduce_sum(tf.square(kernel), axis = 0, keep_dims = True)+1e-10)#(1, 10)
        logits = tf.matmul(bottle_neck, kernel)#(batch_size, 10)
        bottle_neck_len = tf.sqrt(tf.reduce_sum(tf.square(bottle_neck), axis = 1, keep_dims = True)+1e-10)#(batch_size, 1)
        cos_theta = logits/(bottle_neck_len*kernel_len)#(batch_size, 10) 表征了每个feature与对应权重的夹角
        phi = cos_theta - m 
        label_onehot = tf.one_hot(label_batch, nrof_classes)
        adjust_theta = s * tf.where(tf.equal(label_onehot,1), phi, cos_theta)       
        
        '''
        If want to weighted loss of standard softmax and AM-softmax:
        
        f = 1.0/(1.0+lamb)
        ff= 1.0-f
        return f*adjust_theta + ff*cos_theta
        
        '''
        
        return adjust_theta