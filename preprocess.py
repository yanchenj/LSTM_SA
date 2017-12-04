import os
import numpy as np
from random import *
from gensim.models import Word2Vec
from nltk.corpus import movie_reviews


def preprocess():
	train_data, train_label = [],[]
	model = Word2Vec(movie_reviews.sents())

	path = os.getcwd()+"/aclImdb/train/pos/"
	for filename in os.listdir(path):
		pos_file = open("aclImdb/train/pos/"+filename, 'r')
		for line in pos_file.readlines():
			review = []
			for word in line.split():
				if word in model.wv.vocab:
					review.append(model.wv[word])
			train_data.append(review)
			train_label.append((0,1))
		pos_file.close()

	path = os.getcwd()+"/aclImdb/train/neg/"
	for filename in os.listdir(path):
		neg_file = open("aclImdb/train/neg/"+filename, 'r')
		for line in neg_file.readlines():
			review = []
			for word in line.split():
				if word in model.wv.vocab:
					review.append(model.wv[word])
			train_data.append(review)
			train_label.append((1,0))
		neg_file.close()

	print len(train_data),len(train_label)
	x_train = np.array([np.array(xi) for xi in train_data])
	y_train = np.array(train_label)
	print x_train.shape, x_train[0].shape, x_train[1].shape
	print y_train.shape, y_train[0].shape, y_train[1].shape
	np.save('train_data.npy', x_train)
	np.save('train_label.npy', y_train)

	test_data, test_label = [],[]

	path = os.getcwd()+"/aclImdb/test/pos/"
	for filename in os.listdir(path):
		pos_file = open("aclImdb/test/pos/"+filename, 'r')
		for line in pos_file.readlines():
			review = []
			for word in line.split():
				if word in model.wv.vocab:
					review.append(model.wv[word])
			test_data.append(review)
			test_label.append((0,1))
		pos_file.close()

	path = os.getcwd()+"/aclImdb/test/neg/"
	for filename in os.listdir(path):
		neg_file = open("aclImdb/test/neg/"+filename, 'r')
		for line in neg_file.readlines():
			review = []
			for word in line.split():
				if word in model.wv.vocab:
					review.append(model.wv[word])
			test_data.append(review)
			test_label.append((0,1))
		neg_file.close()

	print 'finish loading data'
	print len(test_data),len(test_label)
	x_test = np.array([np.array(xi) for xi in test_data])
	y_test = np.array(test_label)
	np.save('test_data.npy', x_test)
	np.save('test_label.npy', y_test)
	return [np.asarray(x_train),np.asarray(y_train),np.asarray(x_test),np.asarray(y_test)]
