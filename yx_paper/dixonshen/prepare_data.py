# coding = utf-8
from yx_paper.dixonshen.train_prepare import load_features

feature_path1 = './source/res0906_count_over_5.txt'
feature_path2 = './source/res0906.txt'
train_words, freq, rw_score, label = load_features(feature_path1)
train_words1, freq1, rw_score1, label1 = load_features(feature_path2)
training_data = []
with open('training_data.txt', 'w', encoding='utf-8') as f:
	for i in range(len(train_words1)):
		if label1[i] == '否' and int(freq1[i]) <= 5:
			temp = '%.6f' % float(rw_score1[i])
			training_data.append(train_words1[i] + ' ' + freq1[i] + ' ' + str(temp) + ' ' + label1[i])
	for i in range(len(train_words)):
		if label[i] == '是' or int(freq[i]) >= 5000:
			temp = '%.6f' % float(rw_score[i])
			training_data.append(train_words[i] + ' ' + freq[i] + ' ' + str(temp) + ' ' + '是')
	for record in training_data:
		print(record)
		f.write(record + '\n')
print("训练数据（词量）: " + len(training_data).__str__())
