import os
from random import *
from gensim.models import Word2Vec
from nltk.corpus import movie_reviews
			
if __name__ == "__main__":
	pos_data, neg_data = [],[]

	path = os.getcwd()+"/train/small_pos/"
	for filename in os.listdir(path):
		pos_file = open("train/small_pos/"+filename, 'r')
		pos_data = pos_data + pos_file.readlines()
		pos_file.close()

	path = os.getcwd()+"/train/small_neg/"
	for filename in os.listdir(path):
		neg_file = open("train/small_neg/"+filename, 'r')
		neg_data = neg_data + neg_file.readlines()
		neg_file.close()

	p_idx, n_idx = 0,0
	p_len, n_len = len(pos_data), len(neg_data)

	train_data, train_label = [],[]
	model = Word2Vec(movie_reviews.sents())

	while p_idx < p_len and n_idx < n_len:
		if randint(0,1) == 0:
			for word in neg_data[n_idx].split():
				if word in model.wv.vocab:
					train_data.append(model.wv[word])
					train_label.append(0)
			n_idx = n_idx + 1
		else:
			for word in pos_data[p_idx].split():
				if word in model.wv.vocab:	
					train_data.append(model.wv[word])
					train_label.append(1)
			p_idx = p_idx + 1

	if p_idx < p_len:
		for i in xrange(p_idx,p_len):
			for word in pos_data[i].split():
				if word in model.wv.vocab:
					train_data.append(model.wv[word])
					train_label.append(1)
	if n_idx < n_len:
		for i in xrange(n_idx,n_len):
			for word in neg_data[i].split():
				if word in model.wv.vocab:
					train_data.append(model.wv[word])
					train_label.append(0)
	print train_data