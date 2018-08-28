import tensorflow as tf

def AM_logits_compute(embeddings, label_batch, args, nrof_classes):
    
    '''
    loss head proposed in paper:<Additive Margin Softmax for Face Verification>
    link: https://arxiv.org/abs/1801.05599

    embeddings : normalized embedding layer of Facenet, it's normalized value of output of resface
    label_batch : ground truth label of current training batch
    args:         arguments from cmd line
    nrof_classes: number of classes
    '''
    m = 0.35
    s = 30

    with tf.name_scope('AM_logits'):
        kernel = tf.get_variable(name='kernel',dtype=tf.float32,shape=[args.embedding_size,nrof_classes],initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-10, name='kernel_norm')
        cos_theta = tf.matmul(embeddings, kernel_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1,1) # for numerical steady
        phi = cos_theta - m 
        label_onehot = tf.one_hot(label_batch, nrof_classes)
        adjust_theta = s * tf.where(tf.equal(label_onehot,1), phi, cos_theta)
        
        return adjust_theta
