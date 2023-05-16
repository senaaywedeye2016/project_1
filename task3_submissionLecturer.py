# -*- coding: utf-8 -*-
"""task3_submission.ipynb
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, codecs, json, math, time, warnings, re, logging
warnings.simplefilter( action='ignore', category=FutureWarning )

import logging
import tensorflow as tf
import absl.logging
import nltk, numpy, scipy, sklearn, sklearn_crfsuite, sklearn_crfsuite.metrics

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logging started')


import sys, codecs, json, math, time, warnings
warnings.simplefilter( action='ignore', category=FutureWarning )

import nltk, scipy, sklearn, sklearn_crfsuite, eli5
from sklearn.metrics import make_scorer
from collections import Counter
import matplotlib.pyplot as plt
from IPython.display import display    
import eli5

import logging
import tensorflow as tf

import absl.logging
############################# My Code starts from here###########################

# chapter_file = input("Enter the path of the Chapter: ")
# ontonotes_file = input("Enter the path of the ontonotes_file: ")

formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s %(asctime)s] %(message)s')
absl.logging.get_absl_handler().setFormatter(formatter)
absl.logging._warn_preinit_stderr = False
logger = tf.get_logger()
logger.setLevel(logging.INFO)

ontonotes_file = '/data/COMP3225/data/ontonotes_parsed.json'
chapter_file = '/data/COMP3225/data/eval_chapter.txt'


def create_dataset( max_files = None ) :
	dataset_file = ontonotes_file
    
	# load parsed ontonotes dataset
	readHandle = codecs.open( dataset_file, 'r', 'utf-8', errors = 'replace' )
	str_json = readHandle.read()
	readHandle.close()
	dict_ontonotes = json.loads( str_json )

	# make a training and test split
	list_files = list( dict_ontonotes.keys() )
	if len(list_files) > max_files :
		list_files = list_files[ :max_files ]
	nSplit = math.floor( len(list_files)*0.9 )
	list_train_files = list_files[ : nSplit ]
	list_test_files = list_files[ nSplit : ]

	# sent = (tokens, pos, IOB_label)
	list_train = []
	for str_file in list_train_files :
		for str_sent_index in dict_ontonotes[str_file] :
			# ignore sents with non-PENN POS tags
			if 'XX' in dict_ontonotes[str_file][str_sent_index]['pos'] :
				continue
			if 'VERB' in dict_ontonotes[str_file][str_sent_index]['pos'] :
				continue

			list_entry = []

			# compute IOB tags for named entities (if any)
			ne_type_last = None
			for nTokenIndex in range(len(dict_ontonotes[str_file][str_sent_index]['tokens'])) :
				strToken = dict_ontonotes[str_file][str_sent_index]['tokens'][nTokenIndex]
				strPOS = dict_ontonotes[str_file][str_sent_index]['pos'][nTokenIndex]
				ne_type = None
				if 'ne' in dict_ontonotes[str_file][str_sent_index] :
					dict_ne = dict_ontonotes[str_file][str_sent_index]['ne']
					if not 'parse_error' in dict_ne :
						for str_NEIndex in dict_ne :
							if nTokenIndex in dict_ne[str_NEIndex]['tokens'] :
								ne_type = dict_ne[str_NEIndex]['type']
								break
				if ne_type != None :
					if ne_type == ne_type_last :
						strIOB = 'I-' + ne_type
					else :
						strIOB = 'B-' + ne_type
				else :
					strIOB = 'O'
				ne_type_last = ne_type

				list_entry.append( ( strToken, strPOS, strIOB ) )

			list_train.append( list_entry )

	list_test = []
	for str_file in list_test_files :
		for str_sent_index in dict_ontonotes[str_file] :
			# ignore sents with non-PENN POS tags
			if 'XX' in dict_ontonotes[str_file][str_sent_index]['pos'] :
				continue
			if 'VERB' in dict_ontonotes[str_file][str_sent_index]['pos'] :
				continue

			list_entry = []

			# compute IOB tags for named entities (if any)
			ne_type_last = None
			for nTokenIndex in range(len(dict_ontonotes[str_file][str_sent_index]['tokens'])) :
				strToken = dict_ontonotes[str_file][str_sent_index]['tokens'][nTokenIndex]
				strPOS = dict_ontonotes[str_file][str_sent_index]['pos'][nTokenIndex]
				ne_type = None
				if 'ne' in dict_ontonotes[str_file][str_sent_index] :
					dict_ne = dict_ontonotes[str_file][str_sent_index]['ne']
					if not 'parse_error' in dict_ne :
						for str_NEIndex in dict_ne :
							if nTokenIndex in dict_ne[str_NEIndex]['tokens'] :
								ne_type = dict_ne[str_NEIndex]['type']
								break
				if ne_type != None :
					if ne_type == ne_type_last :
						strIOB = 'I-' + ne_type
					else :
						strIOB = 'B-' + ne_type
				else :
					strIOB = 'O'
				ne_type_last = ne_type

				list_entry.append( ( strToken, strPOS, strIOB ) )

			list_test.append( list_entry )

	return list_train, list_test
	
def sent2features(sent, word2features_func = None):
	return [word2features_func(sent, i) for i in range(len(sent))]

def sent2labels(sent):
	return [label for token, postag, label in sent]

def sent2tokens(sent):
	return [token for token, postag, label in sent]

def print_F1_scores( micro_F1 ) :
	for label in micro_F1 :
		logger.info( "%-15s -> f1 %0.2f ; prec %0.2f ; recall %0.2f" % ( label, micro_F1[label]['f1-score'], micro_F1[label]['precision'], micro_F1[label]['recall'] ) )

def print_transitions(trans_features):
	for (label_from, label_to), weight in trans_features:
		logger.info( "%-15s -> %-15s %0.6f" % (label_from, label_to, weight) )

def print_state_features(state_features):
	for (attr, label), weight in state_features:
		logger.info( "%0.6f %-15s %s" % (weight, label, attr) )
  
def task2_word2features(sent, i):

	word = sent[i][0]
	postag = sent[i][1]

	features = {
		'word' : word,
		'postag': postag,

		# token shape
		'word.lower()': word.lower(),
		'word.isupper()': word.isupper(),
		'word.istitle()': word.istitle(),
		'word.isdigit()': word.isdigit(),

		# token suffix
		'word.suffix': word.lower()[-3:],

		# POS prefix
		'postag[:2]': postag[:2],
	}
	if i > 0:
		word_prev = sent[i-1][0]
		postag_prev = sent[i-1][1]
		features.update({
			'-1:word.lower()': word_prev.lower(),
			'-1:postag': postag_prev,
			'-1:word.lower()': word_prev.lower(),
			'-1:word.isupper()': word_prev.isupper(),
			'-1:word.istitle()': word_prev.istitle(),
			'-1:word.isdigit()': word_prev.isdigit(),
			'-1:word.suffix': word_prev.lower()[-3:],
			'-1:postag[:2]': postag_prev[:2],
		})
	else:
		features['BOS'] = True

	if i < len(sent)-1:
		word_next = sent[i+1][0]
		postag_next = sent[i+1][1]
		features.update({
			'+1:word.lower()': word_next.lower(),
			'+1:postag': postag_next,
			'+1:word.lower()': word_next.lower(),
			'+1:word.isupper()': word_next.isupper(),
			'+1:word.istitle()': word_next.istitle(),
			'+1:word.isdigit()': word_next.isdigit(),
			'+1:word.suffix': word_next.lower()[-3:],
			'+1:postag[:2]': postag_next[:2],
		})
	else:
		features['EOS'] = True

	return features
import sklearn_crfsuite
from sklearn_crfsuite import metrics

def task3_train_crf_model(X_train, Y_train, max_iter, labels):
    # Train CRF model using L1 regularization of 200
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=200,
        c2=0.1,
        max_iterations=max_iter,
        all_possible_transitions=True,
    )
    crf.fit(X_train, Y_train)
    
    # Compute F1 score
    y_pred = crf.predict(X_train)
    f1_score = metrics.flat_f1_score(Y_train, y_pred, average='weighted', labels=labels)
    
    print("CRF model trained with F1 score:", f1_score)
    return crf
max_files = 500 
max_iter = 30


	# make a dataset from english NE labelled ontonotes sents
train_sents, test_sents = create_dataset( max_files = max_files )
logger.info( '# training sents = ' + str(len(train_sents)) )
logger.info( '# test sents = ' + str(len(test_sents)) )

	# print example sent (1st sent)
logger.info( '' )
logger.info( 'Example training sent annotated with IOB tags  = ' + repr(train_sents[0]) )

	# create feature vectors for every sent
X_train = [sent2features(s, word2features_func = task2_word2features) for s in train_sents]
Y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s, word2features_func = task2_word2features) for s in test_sents]
Y_test = [sent2labels(s) for s in test_sents]

# create feature vectors for every sent
X_train = [sent2features(s, word2features_func = task2_word2features) for s in train_sents]
Y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s, word2features_func = task2_word2features) for s in test_sents]
Y_test = [sent2labels(s) for s in test_sents]

# get the label set
set_labels = set([])
for data in [Y_train,Y_test] :
  for n_sent in range(len(data)) :
    for str_label in data[n_sent] :
      set_labels.add( str_label )
      labels = list( set_labels )
logger.info( '' )
logger.info( 'labels = ' + repr(labels) )






#################################################################################



def exec_ner( file_chapter = None, ontonotes_file = None ) :
  
  # CHANGE CODE BELOW TO TRAIN A CRF NER MODEL TO TAG THE CHAPTER OF TEXT (task 3)
  # Train CRF model
  crf = task3_train_crf_model( X_train, Y_train, max_iter, labels )


	
  
	# Input >> www.gutenberg.org sourced plain text file for a chapter of a book
	# Output >> ne.json = { <ne_type> : [ <phrase>, <phrase>, ... ] }

 

  import nltk
  nltk.download('punkt')
  # Load the text file
  with open(file_chapter, 'r') as f:
    text = f.read()



  # Tokenize the text using word_tokenize() function from nltk
  chapter = nltk.word_tokenize(text)

	# hardcoded output to show exactly what is expected to be serialized (you should change this)
	# only the allowed types for task 3 DATE, CARDINAL, ORDINAL, NORP will be serialized
  norp_entities = []
  cardinal_entities = []
  ordinal_entities = []
  date_entities = []

  # Predict the labels for the input text
  pred_labels = crf.predict(chapter)

  # Extract the entities of type 'PERSON' and 'GPE'
  N_entities = []
  O_entities = []
  C_entities = []
  D_entities = []
  for i, sent in enumerate(chapter):
      for j, token in enumerate(sent):
          label = pred_labels[i][j]
          if label == 'B-NORP':
              norp_entities.append({'label': label, 'word': X_test[i][j]['word'], 'postag': X_test[i][j]['postag']})
          if label == 'B-ORDINAL':
              ordinal_entities.append({'label': label, 'word': X_test[i][j]['word'], 'postag': X_test[i][j]['postag']})
          if label == 'B-Cardinal':
              cardinal_entities.append({'label': label, 'word': X_test[i][j]['word'], 'postag': X_test[i][j]['postag']})
          if label == 'B-DATE':
              date_entities.append(X_test[i][j]["word"])

  # Print the entities

  print("NORP:")
  for entity in norp_entities:
      print('"',entity,'"')

  print("CARDINAL:")
  for entity in cardinal_entities:
      print('"',entity,'"')

  print("ORDINAL:")
  for entity in ordinal_entities:
      print('"',entity,'"')

  print("DATE:")
  for entity in date_entities:
      print('"',entity,'"')
  import json

  # Create dictionaries for each class
  norp_dict = {'NORP': norp_entities}
  cardinal_dict = {'CARDINAL': cardinal_entities}
  ordinal_dict = {'ORDINAL': ordinal_entities}
  date_dict = {'DATE': date_entities}

  # Append dictionaries to a list
  data = [norp_dict, cardinal_dict, ordinal_dict, date_dict]

  # Save list to JSON file with each dictionary on a separate line
  with open('entities.json', 'w') as f:
      for d in data:
          json.dump(d, f)
          f.write('\n')
# DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

	# FILTER NE dict by types required for task 3
	listAllowedTypes = [ 'DATE', 'CARDINAL', 'ORDINAL', 'NORP' ]
	listKeys = list( dictNE.keys() )
	for strKey in listKeys :
		for nIndex in range(len(dictNE[strKey])) :
			dictNE[strKey][nIndex] = dictNE[strKey][nIndex].strip().lower()
		if not strKey in listAllowedTypes :
			del dictNE[strKey]

	# write filtered NE dict
	writeHandle = codecs.open( 'ne.json', 'w', 'utf-8', errors = 'replace' )
	strJSON = json.dumps( dictNE, indent=2 )
	writeHandle.write( strJSON + '\n' )
	writeHandle.close()


exec_ner( chapter_file, ontonotes_file )

if __name__ == '__main__':
	if len(sys.argv) < 4 :
		raise Exception( 'missing command line args : ' + repr(sys.argv) )
	ontonotes_file = sys.argv[1]
	book_file = sys.argv[2]
	chapter_file = sys.argv[3]

	logger.info( 'ontonotes = ' + repr(ontonotes_file) )
	logger.info( 'book = ' + repr(book_file) )
	logger.info( 'chapter = ' + repr(chapter_file) )

	# DO NOT CHANGE THE CODE IN THIS FUNCTION

	exec_ner( chapter_file, ontonotes_file )



