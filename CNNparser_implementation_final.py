import tensorflow as tf
import numpy as np


class CNNparser(object):
    """
    
    """
    def __init__(
      self, ns, nq, nl,sequence_length,wordattr_num, vocab_size,pos_size,dependency_label_size,
      filter_sizes,num_filters,word_embedding_size, pos_embedding_size, label_embedding_size,
      dl_embedding_size,
      stack_position_em_size,queue_position_em_size,dlp_embedding_size,
      l2_reg_lambda=0.0):
        
        self.input_x = tf.placeholder(tf.int32, [None,3,sequence_length,wordattr_num], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, 3], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")


        # Word Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            N=shape(self.input_x)[0]
            sequence_len=shape(self.input_x)[-2]
            #embedding table for words
            Ww = tf.Variable(
                tf.random_uniform([vocab_size, word_embedding_size], -1.0, 1.0),
                name="Ww")
            #embedding table for POS labels
            Wl= tf.Variable(
                tf.random_uniform([pos_size, pos_embedding_size], -1.0, 1.0),
                name="Wl")
            #embedding table for positions in stack
            Wsp=tf.Variable(
                tf.random_uniform([ns, stack_position_em_size], -1.0, 1.0),
                name="Wsp")
            #embedding table for positions in queue
            Wqp=tf.Variable(
                tf.random_uniform([nq, queue_position_em_size], -1.0, 1.0),
                name="Wqp")
            Wdl= tf.Variable(
                tf.random_uniform([dependency_label_size, dl_embedding_size], -1.0, 1.0),
                name="Wdl")
            Wlp=tf.Variable(
                tf.random_uniform([nl, dlp_embedding_size], -1.0, 1.0),
                name="Wlp")
            
            stack_total_embedding_size=word_embedding_size+pos_embedding_size+stack_position_em_size
            #initialize stack embedding
            stack_embedding=tf.ones([N,0, stack_total_embedding_size], tf.int32)
            
            queue_total_embedding_size=word_embedding_size+pos_embedding_size+queue_position_em_size
            #initialize queue embedding
            queue_embedding=tf.ones([N,0, queue_total_embedding_size], tf.int32)

            arc_set_total_embedding_size=word_embedding_size+pos_embedding_size+dlp_embedding_size
            +dl_embedding_size
            #initialize arc_set embedding
            arc_set_embedding=tf.ones([N,0, arc_set_total_embedding_size], tf.int32)
            
            for i in range(1,ns+1):

                #Get the ith word from stacks
                stack_word=self.input_x[:,1:2,sequence_len-i:sequence_len-i+1,1:2]
                stack_word=tf.reshape(stack_word,[N,1])
                #Create the embedding for the ith word in stacks
                stack_word_embedded = tf.nn.embedding_lookup(Ww,stack_word)
                #Get the ith POS label from stacks
                stack_label=self.input_x[:,1:2,sequence_len-i:sequence_len-i+1,3:4]
                stack_label=tf.reshape(stack_label,[N,1])
                #Create the embedding for the ith word in stacks
                stack_label_embedded = tf.nn.embedding_lookup(Wl,stack_label)
                #Get the position of word in stacks
                stack_position=tf.ones([N, 1], tf.int32)*i
                #Create the embedding for the ith position in stacks
                stack_position_embedded = tf.nn.embedding_lookup(Wsp,stack_postition)
                #Concatenate the embeddings of word, POS label and position of each single element
                stack_single_embedding=tf.concat(2,[stack_word_embedded,stack_label_embedded,
                                                    stack_position_embedded])
                #Combine the embeddings of the first ns elements in the stack
                stack_embedding=tf.concat(1,[stack_embedding,stack_single_embedding])
            #implementing the embedding for the first ns elemets n the queue
            for i in range(1,ns+1):

                
                queue_word=self.input_x[:,0:1,i-1:i,1:2]
                queue_word=tf.reshape(queue_word,[N,1])
                queue_word_embedded = tf.nn.embedding_lookup(Ww,queue_word)
                queue_label=self.input_x[:,0:1,i-1:i,3:4]
                queue_label=tf.reshape(queue_label,[N,1])
                queue_label_embedded = tf.nn.embedding_lookup(Wl,queue_label)
                queue_position=tf.ones([N, 1], tf.int32)*i
                queue_position_embedded = tf.nn.embedding_lookup(Wqp,queue_postition)
                queue_single_embedding=tf.concat(2,[queue_word_embedded,queue_label_embedded,
                                                    queue_position_embedded])
                queue_embedding=tf.concat(1,[queue_embedding,queue_single_embedding])
                
            
            
            for i in range(1,nl+1):
                '''Get the modifier word for the ith dependency arc (in a particular arc,
                the first element is the arc label, the second is the word index of
                the modifier, while the third is the word itself.)'''
                modifier_word=self.input_x[:,2:3,sequence_len-i:sequence_len-i+1,2:3]
                modifier_word=tf.reshape(modifier_word,[N,1])
                modifier_word_embedded = tf.nn.embedding_lookup(Ww,modifier_word)
                modifier_label=self.input_x[:,2:3,i-1:i,4:5]
                modifier_label=tf.reshape(modifier_label,[N,1])
                modifier_label_embedded = tf.nn.embedding_lookup(Wl,modifier_label)
                label_position=tf.ones([N, 1], tf.int32)*i
                label_position_embedded = tf.nn.embedding_lookup(Wlp,queue_postition)
                dependency_label=self.input_x[:,2:3,sequence_len-i:sequence_len-i+1,0:1]
                dependency_label=tf.reshape(dependency_label,[N,1])
                dependency_label_embedded = tf.nn.embedding_lookup(Wdl,modifier_label)
                arc_set_single_embedding=tf.concat(2,[modifier_word_embedded,modifier_label_embedded,
                                                    label_position_embedded,dependency_label_embedded])
                arc_set_embedding=tf.concat(1,[arc_set_embedding,arc_set_single_embedding])

            self.stack_embedding_expanded = tf.expand_dims(stack_embedding, -1)
            self.queue_embedding_expanded = tf.expand_dims(queue_embedding, -1)
            self.arc_set_embedding_expanded=tf.expand_dims(arc_set_embedding, -1)

                
                
                
                
                
                
                
                
            
            
            
        
        

        # Create a convolution + maxpool layer for each filter size
        stack_pooled_outputs = []
        queue_pooled_outputs=[]
        arc_set_pooled_outputs=[]
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                stack_filter_shape = [filter_size, stack_total_embedding_size, 1, num_filters]
                queue_filter_shape=[filter_size, queue_total_embedding_size, 1, num_filters]
                arc_set_filter_shape=[filter_size, arc_set_total_embedding_size, 1, num_filters]
                Ws = tf.Variable(tf.truncated_normal(stack_filter_shape, stddev=0.1), name="Ws")
                bs = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="bs")
                Wq = tf.Variable(tf.truncated_normal(queue_filter_shape, stddev=0.1), name="Wq")
                bq = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="bq")
                Wa = tf.Variable(tf.truncated_normal(arc_set_filter_shape, stddev=0.1), name="Wa")
                ba = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="ba")
                
                
                conv_s = tf.nn.conv2d(
                    self.stack_embedding_expanded,
                    Ws,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_s")
                # Apply nonlinearity
                h_s = tf.nn.relu(tf.nn.bias_add(conv_s, bs), name="relu_s")
                # Maxpooling over the outputs
                pooled_s = tf.nn.max_pool(
                    h_s,
                    ksize=[1, ns - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool_s")
                stack_pooled_outputs.append(pooled_s)

                conv_q = tf.nn.conv2d(
                    self.queue_embedding_expanded,
                    Wq,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_q")
                # Apply nonlinearity
                h_q = tf.nn.relu(tf.nn.bias_add(conv_q, bq), name="relu_q")
                # Maxpooling over the outputs
                pooled_q = tf.nn.max_pool(
                    h_q,
                    ksize=[1, nq - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool_q")
                queue_pooled_outputs.append(pooled_q)

                conv_a = tf.nn.conv2d(
                    self.arc_set_embedding_expanded,
                    Wa,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_a")
                # Apply nonlinearity
                h_a = tf.nn.relu(tf.nn.bias_add(conv_a, ba), name="relu_a")
                # Maxpooling over the outputs
                pooled_a = tf.nn.max_pool(
                    h_a,
                    ksize=[1, nl - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool_a")
                arc_set_pooled_outputs.append(pooled_a)

                
                

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_s = tf.concat(3, stack_pooled_outputs)
        self.h_pool_s_flat = tf.reshape(self.h_pool_s, [-1, num_filters_total])

        self.h_pool_q = tf.concat(3, queue_pooled_outputs)
        self.h_pool_q_flat = tf.reshape(self.h_pool_q, [-1, num_filters_total])

        self.h_pool_a = tf.concat(3, arc_set_pooled_outputs)
        self.h_pool_a_flat = tf.reshape(self.h_pool_a, [-1, num_filters_total])
        #as there are num_filter_total filters for queue, stack and arc_set respectively,
        #there are num_filters_total*3 filters and therefore features for each configuration
        overall_num_filters_total=num_filters_total*3

        self.h_pool_overall=tf.concat(1,[self.h_pool_s_flat,self.h_pool_q_flat,self.h_pool_a_flat])
        

        

        

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_overall, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[overall_num_filters_total, 3],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[3]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            