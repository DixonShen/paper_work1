from utils import *
from sklearn.cross_validation import KFold
from load_word2vec import *
from v3.v3_utils import *
from tensorflow.contrib import learn

from yx_paper.dixonshen.train_prepare import *
from yx_paper.train_cnn import TrainCNN_YX


# 读取训练数据及词频，RW特征
training_data_path = './dixonshen/training_data.txt'
train_words, freq, rw_score, label = load_training_data(training_data_path)
print(train_words)

# 加载vector.bin文件
print('读取vector.bin文件')
vector_path = './dixonshen/source/sougou_vectors.bin'
vocab, embedding = load_from_binary(vector_path)
vocab_size, embedding_dim = embedding.shape
print("embedding_dim:", embedding_dim)
print("shape:", embedding.shape)
print("Loading succeed!")

whether_word2vec = True
whether_tf = True
max_sample_length = 1

# 这里设定为max_sample_length,因为下面函数,if samples are longer,they will be trimmed, if shorter-padding
vocab_processor = learn.preprocessing.VocabularyProcessor(max_sample_length)

if whether_word2vec:
	print('Transforming samples to matrix, preparing data for train:')
	w_train_raw = sample2index_matrix2(train_words, vocab)
	w_train = np.array(makePaddedList_index(max_sample_length, w_train_raw, 1))  # should be np.array() but list
	# w_train = np.array(sample2index_matrix(taj_contents, vocab, max_sample_length))   # should be np.array() but list
	print('w_train shape:', w_train.shape)
	print(w_train[0])
	print(w_train[1])
else:
	print("Building w_train according word2vec vocabulary:")
	# load data
	# f_titles, f_authors, f_journals = load_data_not_word2vec()
	# x_text = f_titles + f_authors + f_journals

	w_train_raw = word2id_vocab2(train_words, vocab)
	w_train = np.array(makePaddedList_index(max_sample_length, w_train_raw, 1))
	# 因为这里直接用原始的sample来构建矩阵,所以还需要padding,"<p> ==> 1"

	print(w_train.shape)
	print(w_train[0])
	print(w_train[1])

# 构建词频特征
tf_dic = build_freq_dic()
tf_dic_size = len(tf_dic) + 1
tf_temp = []
for f in freq:
	temp = []
	temp.append(f)
	tf_temp.append(temp)
tf_train = np.array(mapWordToId(tf_temp, tf_dic))
print('tf_train shape: ', tf_train.shape)

# 构建RW特征
rw_dic = build_rw_dic()
rw_dic_size = len(rw_dic) + 1
print("rw_dict_size: ", rw_dic_size)
rw_temp = []
for r in rw_score:
	temp = []
	temp.append(r)
	rw_temp.append(temp)
rw_train = np.array(mapWordToId(rw_temp, rw_dic))
print('rw_train shape: ', rw_train.shape)

y_train, label_dict_size = build_y_train(label)

# shuffle here firstly!
data_size = len(w_train)
shuffle_indices = np.random.permutation(np.arange(data_size))
s_w_train = w_train[shuffle_indices]
s_y_train = y_train[shuffle_indices]
s_tf_train = tf_train[shuffle_indices]
s_rw_train = rw_train[shuffle_indices]
print(s_w_train.shape)
print(s_y_train.shape)
print(s_tf_train.shape)
print(s_rw_train.shape)

# print("max sample length:", max_sample_length)
# print("vocab_size:", vocab_size)


w_train = s_w_train
tf_train = s_tf_train
rw_train = s_rw_train
y_train = s_y_train

print('w_train:')
print(w_train)
print('tf_train:')
print(tf_train)
print('rw_train:')
print(rw_train)
print('y_train')
print(y_train)

# # ===================================
# print("Start to train:")
# print("Initial TrainCNN: ")
train = TrainCNN_YX(whether_word2vec=whether_word2vec,
                    whether_tf=whether_tf,
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,  # 词向量维度,或者embedding的维度
                    sequence_length=max_sample_length,  # padding之后的句子长度
                    vocab_processor=vocab_processor,
                    num_classes=label_dict_size,
                    tf_dict_size=tf_dic_size,
                    rw_dict_size=rw_dic_size,
                    )
# Split train/test set, use 10_fold cross_validation
print("k_fold train:")
k_fold = KFold(len(s_w_train), n_folds=5)
for train_indices, test_indices in k_fold:
	w_tr, w_te = w_train[train_indices], w_train[test_indices]
	tf_tr, tf_te = tf_train[train_indices], tf_train[test_indices]
	rw_tr, rw_te = rw_train[train_indices], rw_train[test_indices]
	y_tr, y_te = y_train[train_indices], y_train[test_indices]

	train.cnn_train(whether_word2vec, whether_tf, embedding, w_tr, w_te, tf_tr, tf_te, rw_tr, rw_te, y_tr, y_te)
