# coding = utf-8

import os
import sys
from decimal import Decimal as D
from decimal import getcontext
import chardet
import numpy as np
from utils import *

from load_word2vec import load_from_binary


def get_file_list(dir, fileList):
	newDir = dir
	if os.path.isfile(dir):
		fileList.append(dir)
	elif os.path.isdir(dir):
		for s in os.listdir(dir):
			# 如果需要忽略某些文件夹，使用以下代码
			# if s == "xxx":
				# continue
			newDir=os.path.join(dir,s)
			get_file_list(newDir, fileList)
	return fileList


def load_features(path):
	file_list = []
	file_list.append(path)
	train_words = []
	freq = []
	rw_score = []
	label = []
	for file in file_list:
		with open(file, encoding='utf-8') as fileTrainRaw:
			count = 0
			for line in fileTrainRaw:
				if count != 0:
					rows = line.split(" ")
					train_words.append(rows[0])
					freq.append(rows[1])
					rw_score.append(rows[2])
					label.append(rows[3].strip())
				count += 1
	return train_words, freq, rw_score, label


# 构造rw score的字典,区间是[0,1],间隔0.000001,一共1000001个.
def build_rw_dic():
	print('构建RW特征词典')
	a = 0.0
	rw_list = ['0.000000']
	for i in range(1000000):
		a += 0.000001
		rw_list.append(str('%.6f' % a))
		# print('%.5f' % a)
	rw_dic = {}
	index = 0
	for j in rw_list:
		index += 1
		rw_dic[j] = index
	print('RW特征词典构建完成')
	return rw_dic


# 构造rw score的字典,区间是[0,1],间隔0.00001,一共100001个.
def build_freq_dic():
	print('构建词频特征词典')
	a = 0
	freq_list = []
	for i in range(100000):
		freq_list.append(str('%d' % a))
		a += 1
	# print('%.5f' % a)
	freq_dic = {}
	index = 0
	for j in freq_list:
		index += 1
		freq_dic[j] = index
	print('词频特征词典构建完成')
	return freq_dic


def build_y_train(labels):
	print("Building label dict:")
	label_list = []
	for l in labels:
		label_list.append(l)
	label_dict = {'是': 1, '否': 0}
	label_dict_size = len(label_dict)
	
	print("Preparing y_train:")
	y_t = mapLabelToId(label_list, label_dict)
	y_train = np.zeros((len(y_t), label_dict_size))
	for i in range(len(y_t)):
		y_train[i][y_t[i]] = 1
	print("Preparing y_train over!")
	return y_train, label_dict_size

# feature_path = './source/res0906_count_over_5.txt'
# feature_path1 = './source/res0906_count_over_5.txt'
# feature_path2 = './source/res0906.txt'
# train_words, freq, rw_score, label = load_features(feature_path1)
# train_words1, freq1, rw_score1, label1 = load_features(feature_path2)
# training_data = []
# with open('training_data.txt', 'w', encoding='utf-8') as f:
# 	for i in range(len(train_words1)):
# 		if label1[i] == '否' and int(freq1[i]) == 1:
# 			temp = '%.6f' % float(rw_score1[i])
# 			training_data.append(train_words1[i] + ' ' + freq1[i] + ' ' + str(temp) + ' ' + label1[i])
# 	for i in range(len(train_words)):
# 		if label[i] == '是' or int(freq[i]) >= 5000:
# 			temp = '%.6f' % float(rw_score[i])
# 			training_data.append(train_words[i] + ' ' + freq[i] + ' ' + str(temp) + ' ' + '是')
# 	for record in training_data:
# 		print(record)
# 		f.write(record + '\n')
# print("训练数据（词量）: " + len(training_data).__str__())

# with open('./source/house_dict.txt', encoding='utf-8') as f:
# 	count = 0
# 	for line in f:
# 		if count != 0:
# 			line = line.strip()
# 			if line in train_words:
# 				print(line)
# 		count += 1
# print(count)

# 找到出现次数等于1的样本
# implicit_count = 0
# reg_rw_score = []
# for i in range(len(train_words)):
# 	if int(freq[i]) == 1 and label[i] == '否':
# 		implicit_count += 1
# 		temp = '%.6f' % float(rw_score[i])
# 		reg_rw_score.append(temp)
# 		print(train_words[i] + ", count: " + freq[i] + ", rw_score: " + str(temp))
# print("implicit count: " + str(implicit_count))


# 读取训练数据
def load_training_data(path):
	print('读取训练数据')
	train_words = []
	freq = []
	rw_score = []
	label = []
	with open(path, encoding='utf-8') as fileTrainRaw:
		count = 0
		for line in fileTrainRaw:
			if count != 0:
				rows = line.split(" ")
				words_list = [rows[0]]
				train_words.append(words_list)
				freq.append(rows[1])
				rw_score.append(rows[2])
				label.append(rows[3].strip())
			count += 1
	print('训练数据读取完成')
	return train_words, freq, rw_score, label

# # 读取训练数据及词频，RW特征
# training_data_path = './training_data.txt'
# train_words, freq, rw_score, label = load_training_data(training_data_path)
# freq_dic = build_freq_dic()
# rw_dic = build_rw_dic()
#
# # 加载vector.bin文件
# print('读取vector.bin文件')
# vector_path = './source/sougou_vectors.bin'
# vocab, embedding = load_from_binary(vector_path)
# vocab_size, embedding_dim = embedding.shape
# print("embedding_dim:", embedding_dim)
# print("shape:", embedding.shape)
# print("Loading succeed!")
#
# whether_word2vec = True
# max_sample_length = 1
#
# if whether_word2vec:
# 	print('Transforming samples to matrix, preparing data for train:')
# 	w_train_raw = sample2index_matrix2(train_words, vocab)
# 	w_train = np.array(makePaddedList_index(max_sample_length, w_train_raw, 1))   # should be np.array() but list
# 	# w_train = np.array(sample2index_matrix(taj_contents, vocab, max_sample_length))   # should be np.array() but list
# 	print('w_train shape:', w_train.shape)
# 	print(w_train[0])
# 	print(w_train[1])
# else:
# 	print("Building w_train according word2vec vocabulary:")
# 	# load data
# 	# f_titles, f_authors, f_journals = load_data_not_word2vec()
# 	# x_text = f_titles + f_authors + f_journals
#
# 	w_train_raw = word2id_vocab2(train_words, vocab)
# 	w_train = np.array(makePaddedList_index(max_sample_length, w_train_raw, 1))
# 	# 因为这里直接用原始的sample来构建矩阵,所以还需要padding,"<p> ==> 1"
#
# 	print(w_train.shape)
# 	print(w_train[0])
# 	print(w_train[1])
