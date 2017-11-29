import glob
from gensim.models import Word2Vec
from nltk.corpus import movie_reviews

if __name__ == "__main__":
	preprocessed_data = []
	outfile = open('preprocess.out', 'w')
	for filename in glob.glob('../acllmdb/train/small_pos/'):
		trainfile = open(filename, 'r')
		train_data = trainfile.readlines()
		for line in train_data:
			outfile.write("1 "+line+"\n")
		trainfile.close()

	for filename in glob.glob('../acllmdb/train/small_neg/'):
		trainfile = open(filename, 'r')
		train_data = trainfile.readlines()
		for line in train_data:
			outfile.write("0 "+line+"\n")
		trainfile.close()
	outfile.close()
	
	# model = Word2Vec(movie_reviews.sents())
	# file = open('preprocess.out','r')
	# label = []
	# data = []
	# for line in file.readlines().split():
	# 	label.append(line[0])
	# 	data.append(model.wv[line[1:]])
	# np.savetxt('data.txt',data)
	# np.savetxt('label.txt',label)

