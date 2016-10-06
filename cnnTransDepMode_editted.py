
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

def shift(queue1, stack1, arc_set1):
     word = queue1.popleft()
     stack1.append(word)

def right_arc(queue1, stack1, arc_set1):
     modifier = stack1.pop()
     head = stack1.pop()
     stack1.append(head)
     label = modifier[7]
     arc = (*label,modifier,*head)
     arc_set1.append(arc)

def left_arc(queue1, stack1, arc_set1):
     head = stack1.pop()
     modifier = stack1.pop()
     stack1.append(head)
     label = modifier[7]
     arc = (label,*modifier,*head)
     arc_set1.append(arc)

def s0_has_other_child(queue1, stack1):
     s0 = stack1[len(stack1)-1]
     result = 0
     for element in stack1:
          if element[6] == s0[0]:
               result = 1 
     for element in queue1:
          if element[6] == s0[0]:
               result = 1
     return result
          

def generate_samples(sentence,training_set):
     stack = []
     queue = deque([])
     arc_set = []

     #push all the words to the queue
     for word in sentence:
          queue.append(word)

     #shfit the first tow words from queue to the stack
     shift(queue, stack, arc_set)

     #process the words in queue and stack
     while len(queue)>0 or len(stack)>=2:
          configuration = (queue, stack, arc_set)

          if len(stack)>=2 and stack[len(stack)-2][6]== stack[len(stack)-1][0]:
               #left arc
               left_arc(queue, stack, arc_set)
               print("left arc:", arc_set[len(arc_set)-1])
               transition = {"left_arc",arc_set[len(arc_set)-1][0]}
               
          
          elif len(stack)>=2 and stack[len(stack)-1][6]== stack[len(stack)-2][0] and not s0_has_other_child(queue, stack):
               #right arc
               right_arc(queue, stack, arc_set)
               print("right arc:", arc_set[len(arc_set)-1])
               transition = {"right_arc",arc_set[len(arc_set)-1][0]}
               

          elif len(queue)>0:
               #shift
               shift(queue, stack, arc_set)
               print("shift:", stack[len(stack)-1])
               transition = {"shift", stack[len(stack)-1][1]}
               

          training_example = (configuration,transition)
          training_set.append(training_example)
         
     #process the root node
     modifier = stack.pop()
     label = modifier[7]
     arc = (label,modifier,[])
     arc_set.append(arc)
     configuration = (queue, stack, arc_set)
     transition = {"right_arc",arc_set[len(arc_set)-1][0]}
     training_example = (configuration,transition)
     training_set.append(training_example)

     #complete the processing
     #print the results
     print("complete")
     print("queue:", queue)
     print("stack:", stack)
     print("arc set:", arc_set)

################
     '''
class embedding:
     def __init__(self,fileName, WordEmbeddingDim = 1000,
                  TagEmbeddingDim = 100,
                  LabelEmbeddingDim = 100):          
          #print "file name: %s" % fileName

          labelList = []
          wordlist = []
          postaglist = []

          with open(fileName,"r") as ins:
               for line in ins:
                    elements = line.split("\t")
                    if len(elements) == 10:
                         wordlist.append(elements[1])
                         postaglist.append(elements[3])
                         labelList.append(elements[7])

          self.wordlist = list(set(wordlist))
          self.postaglist = list(set(postaglist))
          self.labelList = list(set(labelList))
          
          self.label_size = len(labelList)
          self.word_size = len(wordlist)
          self.postag_size = len(postaglist)


          print "number of words: %s, number of POS tages: %s, number of dependency labels: %s" \
               % (self.word_size, self.postag_size, self.label_size)

          self.word_Embedding = tf.Variable( \
               tf.random_uniform([self.word_size, WordEmbeddingDim], -1.0, 1.0), \
               name="Embedding_word")

          self.tag_Embedding = tf.Variable( \
               tf.random_uniform([self.postag_size, TagEmbeddingDim], -1.0, 1.0), \
               name="Embedding_tag")

          self.label_Embedding = tf.Variable( \
               tf.random_uniform([self.label_size, LabelEmbeddingDim], -1.0, 1.0), \
               name="Embedding_label")
          
          #initialize the variable
          init_op = tf.initialize_all_variables()
          #run the graph
          with tf.Session() as sess:
               sess.run(init_op) #execute init_op
               #print the random values that we sample
               print sess.run(self.word_Embedding)
               print sess.run(self.tag_Embedding)
               print sess.run(self.label_Embedding)

################

class parser:
     def __init__(self, sentence):
          self.sentence = sentence

     def parsing():
          queue = []
          stack = []
          arc_set = []
          queue = sentence[1:(len(sentence)-1)]

class trainingset:

     def __init__(self,train_filename):
          print "trainingset"

class configuration:

     def __init__(self):
          print "configuration"
     
class cnnTransDep:

     def __init__(self, fileName, embedding1,configuration):
          print "file name: %s" % fileName

          
          with open("labelList", 'rb') as f:
               labelList = pickle.load(f)
               label_size = len(labelList)
               print "size of label list: %s" % len(labelList)


          
          Embedding_label = labelEmbedding1.Embedding_label

          index_list = []
          with open(fileName,"r") as ins:
               for line in ins:
                    elements = line.split("\t")
                    if len(elements) == 10:
                         index =  labelList.index(elements[7])
                         index_list.append(index)

          self.label_embedding_matrix = tf.nn.embedding_lookup(Embedding_label, index_list)
          print self.label_embedding_matrix
          
'''



