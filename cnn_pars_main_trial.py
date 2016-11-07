#! /usr/bin/env python

import tensorflow as tf
import numpy as np
from cnn_pars_prepare_training161103 import *
from cnn_pars_embedding import *
from cnn_pars_cnn_module import *
from collections import deque
import pickle

#bro this is a different file use this one not the old ones!
################################################
inputs = {}
options = {}

'''
#obtain the complete list of candidate transitions
inputs["training_file"] = "train.conll"
options["trainsition_list_file"] = "trainsition_list_file"
outputs = obtain_transition_list(inputs,options)
trainsition_list = pickle.load( open( outputs["trainsition_list_file"] , "rb"))
print "transition_list: ", trainsition_list
print "len(transition_list) = ", len(trainsition_list)
print

#prepare the complete list of words, tags, and labels
filename = "train.conll"
outputs=obtain_list_of_words_tags_labels(filename)

print "wordlist:", outputs["wordlist"][:10]
print "tag_list:", outputs["tag_list"]
print "label_list:", outputs["label_list"]
print
print "number of words in the dictionary: ", len(outputs["wordlist"])
print "number of tags in the tag_list: ", len(outputs["tag_list"])
print "number of labels in the label_list: ", len(outputs["label_list"])
print
'''

'''
#################################################
#prepare the training set
trainsition_list = pickle.load(open(  "trainsition_list_file" , "rb"))
inputs["trainsition_list"] = trainsition_list
inputs["filename"] = "train.conll"
options["training_example_file"] = "training_example_file.train"
generate_examples_from_file(inputs,options)
'''

inputs["trainsition_list"] = pickle.load(open(  "trainsition_list_file" , "rb"))
inputs["filename"] = "train.conll"
inputs["sentence_id"] = 25
outputs = generate_example_from_file_one_sentence(inputs, options)
for training_example in outputs["training_set"]:
    show_example(training_example)
training_set=outputs["training_set"]

'''
#load the training set
training_example_file = "training_example_file.train"
print "Loading the training example from file:", training_example_file
print
training_example_file = "training_example_file.train"
training_set = pickle.load( open( training_example_file, "rb" ) )
print "number of training examples: ", len(training_set)
print

for training_example in training_set:
    show_example(training_example)
'''


############################################################

#initialize the embedding matrices
length_embedding={}
length_embedding["stack_word_max_num"] = 100
length_embedding["queue_word_max_num"] = 100
length_embedding["arc_max_num"] = 100

embdding_dim={}
embdding_dim["dim_embedding_word"] = 256
embdding_dim["dim_embedding_tag"] = 128
embdding_dim["dim_embedding_label"] = 129
embdding_dim["dim_embedding_stack_pos"] = 16
embdding_dim["dim_embedding_queue_pos"] = 17
embdding_dim["dim_embedding_art_pos"] = 18
embdding_dim["stack"] = embdding_dim["dim_embedding_word"]+embdding_dim["dim_embedding_tag"]+embdding_dim["dim_embedding_stack_pos"]
embdding_dim["queue"] = embdding_dim["dim_embedding_word"]+embdding_dim["dim_embedding_tag"]+embdding_dim["dim_embedding_queue_pos"]
embdding_dim["arc_set"] = embdding_dim["dim_embedding_word"]+embdding_dim["dim_embedding_tag"]\
                          +embdding_dim["dim_embedding_label"]+embdding_dim["dim_embedding_art_pos"]

Embedding_matrix = init_embedding_matrices(embdding_dim,length_embedding)

'''
list of variables:
Embedding_matrix
filter_matrix
classifier_matrix
'''


inputs={}
options={}
options["length_embedding"] = length_embedding
inputs["Embedding_matrix"] = Embedding_matrix
inputs["training_set"] = training_set


'''
cnn representation and classification of the embeddings
'''
filter_num={}
filter_num["stack"]=200
filter_num["queue"]=250
filter_num["arc_set"]=300
inputs["filter_matrix_stack"]=tf.Variable(tf.truncated_normal([embdding_dim["stack"],filter_num["stack"]], stddev=0.1))
inputs["filter_matrix_queue"]=tf.Variable(tf.truncated_normal([embdding_dim["queue"],filter_num["queue"]], stddev=0.1))
inputs["filter_matrix_arc_set"]=tf.Variable(tf.truncated_normal([embdding_dim["arc_set"],filter_num["arc_set"]], stddev=0.1))
'''
print "inputs[filter_matrix_arc_set]:",inputs["filter_matrix_arc_set"]
print
'''

trainsition_list = pickle.load( open( "trainsition_list_file", "rb"))
inputs["classifier_matrix"]=tf.Variable(tf.truncated_normal([filter_num["stack"]+filter_num["queue"]+filter_num["arc_set"],\
                                                             len(trainsition_list)], stddev=0.1))
#inputs["classifier_bias"]=tf.Variable(tf.truncated_normal([filter_num["stack"]+filter_num["queue"]+filter_num["arc_set"],\
#                                                           len(trainsition_list)], stddev=0.1))

options["lambda_embedding_matrix"] = 0.00001
options["lambda_filter_matrix"] = 0.00001
options["lambda_classifier_matrix"] = 0.00001
options["iteration_num"] = 100

cnn_train_parameters(inputs, options)