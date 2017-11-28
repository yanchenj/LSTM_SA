import os
			
if __name__ == "__main__":
	preprocessed_data = []
	outfile = open('preprocess.out', 'w')
	path = os.getcwd()+"/train/small_pos/"
	for filename in os.listdir(path):
		trainfile = open("train/small_pos/"+filename, 'r')
		train_data = trainfile.readlines()
		for line in train_data:
			outfile.write("1 "+line+"\n")
		trainfile.close()

	for filename in os.listdir(os.getcwd()+"/train/small_neg"):
		trainfile = open("train/small_neg/"+filename, 'r')
		train_data = trainfile.readlines()
		for line in train_data:
			outfile.write("0 "+line+"\n")
		trainfile.close()
	outfile.close()

