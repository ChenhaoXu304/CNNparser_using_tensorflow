import tensorflow as tf
import numpy as np
import pickle
from collections import deque

'''
0: index
1: word
3: POS tag
6: parent index
7: label
'''

def show_example(training_example):
     #print training_example 
     print "stack:",
     for elements in training_example["configuration"]["stack"]:
          print elements[1],
     print " || queue:",
     for elements in training_example["configuration"]["queue"]:
          print elements[1],
     print
     print "arc set:",
     for elemetns in training_example["configuration"]["arc_set"]:
          print elemetns[1][1], "->", elemetns[2][1], " (",  elemetns[0], ") |",
     print
     print "transition:", training_example["transition"]["action"]
     #, "(", training_example["transition"]["label"], ")"
     print 

   
def shift_return_example(queue1, stack1, arc_set1):
     training_example = {}
     #explicitly  copy the current configuration to the training example
     training_example["configuration"] = {}
     training_example["configuration"]["queue"] = list(queue1)
     training_example["configuration"]["stack"] = list(stack1)
     training_example["configuration"]["arc_set"] = list(arc_set1)
     #do the shift
     word = queue1.popleft()
     stack1.append(word)
     #print "shift:", stack1[-1]
     #record the gold transition
     training_example["transition"] = {}
     training_example["transition"]["action"] ="shift"
     training_example["transition"]["label"] = stack1[-1][1]
     '''
     #show the training example for this shift
     show_example(training_example)
     '''
     #return the obtained training example
     return training_example

def right_arc_return_example(queue1, stack1, arc_set1,head_index):
     training_example = {}
     #explicitly  copy the current configuration to the training example
     training_example["configuration"] = {}
     training_example["configuration"]["queue"] = list(queue1)
     training_example["configuration"]["stack"] = list(stack1)
     training_example["configuration"]["arc_set"] = list(arc_set1)
     #do the right_arc
     head = stack1[head_index]
     modifier = stack1.pop()
     
     
     label = modifier[7]
     arc = (label,head,modifier)
     arc_set1.append(arc)
     #print "right arc:", arc_set1[-1]
     #record the gold transition
     training_example["transition"] = {}
     training_example["transition"]["action"] ="right_arc:" + label
     training_example["transition"]["label"] = label
     '''
     #show the training example for this right_arc
     show_example(training_example)
     '''
     #return the obtained training example
     return training_example

def left_arc_return_example(queue1, stack1, arc_set1,modifier_index):
     training_example = {}
     #explicitly  copy the current configuration to the training example
     training_example["configuration"] = {}
     training_example["configuration"]["queue"] = list(queue1)
     training_example["configuration"]["stack"] = list(stack1)
     training_example["configuration"]["arc_set"] = list(arc_set1)
     #do the left_arc
     #added jumping to solve jamming problem for certain sentences
     head = stack1[-1]
     modifier = stack1.pop(modifier_index)
     
     
     label = modifier[7]
     arc = (label,head,modifier)
     arc_set1.append(arc)
     #print "left arc:", arc_set1[-1]
     #record the gold transition
     training_example["transition"] = {}
     training_example["transition"]["action"] ="left_arc:" + label
     training_example["transition"]["label"] = label
     '''
     #show the training example for this left_arc
     show_example(training_example)
     '''
     #return the obtained training example
     return training_example

def process_the_root_return_example(queue1, stack1, arc_set1):
     training_example = {}
     #explicitly  copy the current configuration to the training example
     training_example["configuration"] = {}
     training_example["configuration"]["queue"] = list(queue1)
     training_example["configuration"]["stack"] = list(stack1)
     training_example["configuration"]["arc_set"] = list(arc_set1)
     #process the root node
     modifier = stack1.pop()
     label = modifier[7]
     arc = (label,['0'],modifier) 
     arc_set1.append(arc)
     #print "right arc:", arc_set1[-1]
     #record the gold transition
     training_example["transition"] = {}
     training_example["transition"]["action"] ="right_arc:" + label
     training_example["transition"]["label"] = label
     '''
     #show the training example for this left_arc
     show_example(training_example)
     '''
     #return the obtained training example
     return training_example   

def s0_has_other_children(queue1, stack1):
     s0 = stack1[-1]
     result = 0
     for element in stack1:
          if element[6] == s0[0]:
               result = 1 
     for element in queue1:
          if element[6] == s0[0]:
               result = 1
     return result

def modifier_has_other_children(queue1, stack1,modifier_index):
     s0 = stack1[modifier_index]
     result = 0
     for element in stack1:
          if element[6] == s0[0]:
               result = 1 
     for element in queue1:
          if element[6] == s0[0]:
               result = 1
     return result

def generate_samples_from_sentence(inputs,options):
     sentence = inputs["sentence"]
     training_set = inputs["training_set"]
     stack = []
     queue = deque([])
     arc_set = []

     num_example = len(training_set)

     #push all the words to the queue
     for word in sentence:
          queue.append(word)

     #shfit the first word from queue to the stack
     training_example= shift_return_example(queue, stack, arc_set)
     training_set.append(training_example)

     #process the words in queue and stack
     while len(queue)>0 or len(stack)>=2:
          left_arc_executed=False
          right_arc_executed=False
          if len(stack)>=2:
               for i in range(2,len(stack)+1):
                    if stack[-i][6]== stack[len(stack)-1][0] and not modifier_has_other_children(queue,stack,-i):
                         #left arc
                         training_example = left_arc_return_example(queue, stack, arc_set,-i)
                         training_set.append(training_example)
                         left_arc_executed=True
                         break
          if left_arc_executed:
               continue
          
          if len(stack)>=2:
               for i in range(2,len(stack)+1):
                    
                    if stack[-1][6]==stack[-i][0] and not s0_has_other_children(queue, stack):
               #right arc
                         
                         training_example = right_arc_return_example(queue, stack, arc_set,-i)               
                         training_set.append(training_example)
                         right_arc_executed=True
                         break
          if right_arc_executed:
               continue
                         

          if len(queue)>0:
               #shift
               training_example = shift_return_example(queue, stack, arc_set)
               training_set.append(training_example)
               continue
          
     #process the root node
     process_the_root_return_example(queue, stack, arc_set)
     training_set.append(training_example)

     print "Number of training examples of the sentence:", len(training_set)-num_example
     print "\n"
     
     '''
     for training_example in training_set[-(len(training_set)-num_example)]:
          show_example(training_example)
     '''


def generate_examples_from_file(inputs, options):          
     sentence = []
     sentence_words = []
     training_set = []

     inputs1 = {}
     optioins1 ={}
     inputs1["sentence"] = []
     inputs1["training_set"] = []

     num_sentence = 1

     f = open(inputs["filename"])
     line = f.readline()
     while line:    
          word = line.split("\t")
          if len(word) == 10:
               inputs1["sentence"].append(word)
               sentence_words.append(word[1])          
          else:
               print num_sentence, " - sentence:", sentence_words
               print
               num_sentence=num_sentence+1
               generate_samples_from_sentence(inputs1,optioins1)
               inputs1["sentence"][:] = []
               sentence_words[:] = []
               
          line = f.readline()   
     f.close()

     #code the transition action to binary vectors
     num_transition_list = len(inputs["trainsition_list"])
     inputs["trainsition_list"]
     for example in inputs1["training_set"]:          
          example["transition"]["action_vector"] = [0] * num_transition_list
          index_true_transition = inputs["trainsition_list"].index(example["transition"]["action"])
          example["transition"]["action_vector"][index_true_transition] = 1
          '''
          print example["transition"]["action"], "index: ", index_true_transition
          print "transition vector:", example["transition"]["action_vector"]
          '''

     outfile = open(options["training_example_file"], 'w')
     pickle.dump(inputs1["training_set"], outfile)

     outputs = {}
     outputs["training_set"] = inputs1["training_set"]
     outputs["training_example_file"] = options["training_example_file"]

     for training_example in outputs["training_set"]:
          show_example(training_example)

     print "number of training examples: %d" % len(outputs["training_set"])
     print "complete the process of %s" % inputs["filename"]
     print
     print 

     return outputs

def obtain_list_of_words_tags_labels(filename):
     wordlist = []
     tag_list = []
     label_list = []
     #find a complete list of words
     with open(filename,"r") as ins:
          for line in ins:
               elements = line.split("\t")
               if len(elements) == 10:
                    wordlist.append(elements[1])
                    tag_list.append(elements[3])
                    label_list.append(elements[7])
                    
     wordlist = list(set(wordlist))
     tag_list = list(set(tag_list))
     label_list = list(set(label_list))

     #save the complete list to files
     outfile = open("word_list", 'w')
     pickle.dump(wordlist, outfile)
     outfile = open("tag_list", 'w')
     pickle.dump(tag_list, outfile)
     outfile = open("label_list", 'w')
     pickle.dump(label_list, outfile)

     outputs={}
     outputs["wordlist"]=wordlist
     outputs["tag_list"]=tag_list
     outputs["label_list"]=label_list

     return outputs

def obtain_transition_list(inputs = {}, options = {}):
     '''
     0: index
     1: word
     3: POS tag
     6: parent index
     7: label
     '''
     label_list = []
     ins = open(inputs["training_file"],"r")
     for line in ins:
          elements = line.split("\t")
          if len(elements) == 10:
               label_list.append(elements[7])
     label_list = list(set(label_list))

     trainsition_list = []
     trainsition_list.append("shift")
     for label in label_list:
          trainsition_list.append("right_arc:" + label)
          trainsition_list.append("left_arc:" + label)

     #print options

     outputs = {}
     outputs["trainsition_list"] = trainsition_list
     outputs["trainsition_list_file"] = options["trainsition_list_file"]
     outfile = open(outputs["trainsition_list_file"], 'w')
     pickle.dump(trainsition_list, outfile)         
     
     return outputs

'''
#################old version###############

def shift(queue1, stack1, arc_set1):
     word = queue1.popleft()
     stack1.append(word)

def right_arc(queue1, stack1, arc_set1):
     modifier = stack1.pop()
     head = stack1.pop()
     stack1.append(head)
     label = modifier[7]
     arc = (label,head,modifier)
     arc_set1.append(arc)

def left_arc(queue1, stack1, arc_set1):
     head = stack1.pop()
     modifier = stack1.pop()
     stack1.append(head)
     label = modifier[7]
     arc = (label,head,modifier)
     arc_set1.append(arc)
'''

