# encoding = utf8
from util import *
from Preprocessing import *


# aux
def before_extract_feature_load_data(train_file,test_file):
    train_data = pd.read_csv(train_file,sep='\t', header=None,
                                names=["index", "s1", "s2", "label"])

    test_data = pd.read_csv(test_file, sep='\t', header=None,
                                names=["index", "s1", "s2", "label"])
    return train_data,test_data


# aux
def after_extract_feature_save_data(feature_train,feature_test,col_names,feature_name):
    project.save_features(feature_train, feature_test, col_names, feature_name)


def extract_sentence_length_diff():
    """
    长度差特征
    """
    # step1 定义抽取特征的方式名
    feature_name = 'nlp_sentece_length_diff'

    # step2 载入数据
    train_data ,test_data = before_extract_feature_load_data(train_file=project.preprocessed_data_dir + 'train_0.6_seg.csv',
                                                             test_file=project.preprocessed_data_dir + 'test_0.4_seg.csv')

    feature_train = np.zeros((train_data.shape[0],1),dtype='float64')
    feature_test = np.zeros((test_data.shape[0],1),dtype='float64')

    # 计算两个句子的长度差
    def get_length_diff(s1, s2):
        return 1 - abs(len(s1) - len(s2)) / float(max(len(s1), len(s2)))

    for index,row in train_data.iterrows():
        s1 = row['s1'].strip().split(' ')
        s2 = row['s2'].strip().split(' ')
        diff = get_length_diff(s1,s2)
        feature_train[index] = round(diff,5)

    for index, row in test_data.iterrows():
        s1 = row['s1'].strip().split(' ')
        s2 = row['s2'].strip().split(' ')
        diff = get_length_diff(s1, s2)
        feature_test[index] = round(diff,5)

    # step 3 保存特征：参数有：训练集的特征，测试集的特征，抽取特征的方法的多列特征的列名，抽取特征的方式名
    col_names = [feature_name]
    after_extract_feature_save_data(feature_train, feature_test, col_names, feature_name)


def extract_edit_distance():

    # step1 定义抽取特征的方式名
    feature_name = 'nlp_edit_distance'

    # step2 载入数据
    train_data, test_data = before_extract_feature_load_data(
        train_file=project.data_dir + 'atec_nlp_sim_train_0.6.csv',
        test_file=project.data_dir + 'atec_nlp_sim_test_0.4.csv')

    feature_train = np.zeros((train_data.shape[0], 1), dtype='float64')
    feature_test = np.zeros((test_data.shape[0], 1), dtype='float64')

    # 计算编辑距离
    def get_edit_distance(rawq1, rawq2):
        m, n = len(rawq1) + 1, len(rawq2) + 1
        matrix = [[0] * n for i in range(m)]
        matrix[0][0] = 0
        for i in range(1, m):
            matrix[i][0] = matrix[i - 1][0] + 1
        for j in range(1, n):
            matrix[0][j] = matrix[0][j - 1] + 1
        cost = 0
        for i in range(1, m):
            for j in range(1, n):
                if rawq1[i - 1] == rawq2[j - 1]:
                    cost = 0
                else:
                    cost = 1
                matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + cost)
        return 1 - matrix[m - 1][n - 1] / float(max(len(rawq1), len(rawq2)))

    for index,row in train_data.iterrows():
        s1 = row['s1'].strip()
        s2 = row['s2'].strip()
        edit_distance = get_edit_distance(s1,s2)
        feature_train[index] = round(edit_distance,5)

    for index, row in test_data.iterrows():
        s1 = row['s1'].strip()
        s2 = row['s2'].strip()
        edit_distance = get_edit_distance(s1, s2)
        feature_test[index] = round(edit_distance,5)

    # step 3 保存特征：参数有：训练集的特征，测试集的特征，抽取特征的方法的多列特征的列名，抽取特征的方式名
    col_names = [feature_name]
    after_extract_feature_save_data(feature_train,feature_test,col_names,feature_name)


def extract_ngram(max_ngram = 3):
    '''
    提取ngram特征
    :return:
    '''
    # step1 定义抽取特征的方式名
    feature_name = 'nlp_ngram'

    # step2 载入数据
    train_data, test_data = before_extract_feature_load_data(
        train_file=project.preprocessed_data_dir + 'train_0.6_seg.csv',
        test_file=project.preprocessed_data_dir + 'test_0.4_seg.csv')

    feature_train = np.zeros((train_data.shape[0], max_ngram), dtype='float64')
    feature_test = np.zeros((test_data.shape[0], max_ngram), dtype='float64')

    # 定义n_gram的方法
    def get_ngram(rawq, ngram_value):
        result = []
        for i in range(len(rawq)):
            if i + ngram_value < len(rawq) + 1:
                result.append(rawq[i:i + ngram_value])
        return result

    def get_ngram_sim(q1_ngram, q2_ngram):
        q1_dict = {}
        q2_dict = {}
        for token in q1_ngram:
            if token not in q1_dict:
                q1_dict[token] = 1
            else:
                q1_dict[token] = q1_dict[token] + 1
        q1_count = np.sum([value for key, value in q1_dict.items()])

        for token in q2_ngram:
            if token not in q2_dict:
                q2_dict[token] = 1
            else:
                q2_dict[token] = q2_dict[token] + 1
        q2_count = np.sum([value for key, value in q2_dict.items()])

        # ngram1有但是ngram2没有
        q1_count_only = np.sum([value for key, value in q1_dict.items() if key not in q2_dict])
        # ngram2有但是ngram1没有
        q2_count_only = np.sum([value for key, value in q2_dict.items() if key not in q1_dict])
        # ngram1和ngram2都有的话，计算value的差值
        q1_q2_count = np.sum([abs(value - q2_dict[key]) for key, value in q1_dict.items() if key in q2_dict])
        # ngram1和ngram2的总值
        all_count = q1_count + q2_count
        # print(q1_dict)
        # print(q2_dict)
        # print(q1_count_only)
        # print(q2_count_only)
        # print(q1_q2_count)
        # print(all_count)
        return (1 - float(q1_count_only + q2_count_only + q1_q2_count) / (float(all_count) + 0.00000001))

    for ngram_value in range(max_ngram):
        for index, row in train_data.iterrows():
            s1 = row['s1'].strip()
            s2 = row['s2'].strip()
            ngram1 = get_ngram(s1, ngram_value + 1)
            ngram2 = get_ngram(s2, ngram_value + 1)
            ngram_sim = get_ngram_sim(ngram1, ngram2)
            feature_train[index,ngram_value] = round(ngram_sim,5)

        for index, row in test_data.iterrows():
            s1 = row['s1'].strip()
            s2 = row['s2'].strip()
            ngram1 = get_ngram(s1, ngram_value + 1)
            ngram2 = get_ngram(s2, ngram_value + 1)
            ngram_sim = get_ngram_sim(ngram1, ngram2)
            ngram_sim = get_ngram_sim(ngram1, ngram2)
            feature_test[index, ngram_value] = round(ngram_sim, 5)


    # step 3 保存特征：参数有：训练集的特征，测试集的特征，抽取特征的方法的多列特征的列名，抽取特征的方式名
    col_names = [('{}_{}'.format(feature_name,ngram_value))for ngram_value in range(max_ngram)]
    after_extract_feature_save_data(feature_train,feature_test,col_names,feature_name)


def extract_sentence_diff_same():
    '''
    两个句子的相同和不同的词特征
    '''
    # step1 定义抽取特征的方式名
    feature_name = 'nlp_sentece_diff_some'
    col_num = 6
    # step2 载入数据
    train_data, test_data = before_extract_feature_load_data(
        train_file=project.preprocessed_data_dir + 'train_0.6_seg.csv',
        test_file=project.preprocessed_data_dir + 'test_0.4_seg.csv')

    feature_train = np.zeros((train_data.shape[0],col_num),dtype='float64')
    feature_test = np.zeros((test_data.shape[0],col_num),dtype='float64')

    #统计两个句子的相同和不同
    def get_word_diff(q1, q2):
        set1 = set(q1.split(" "))
        set2 = set(q2.split(" "))
        same_word_len = len(set1 & set2)
        unique_word1_len = len(set1 - set2)
        unique_word2_len = len(set2 - set1)
        word1_len = len(set1)
        word2_len = len(set2)
        avg_len = (word1_len + word2_len) / 2.0
        max_len = max(word1_len, word2_len)
        min_len = min(word1_len, word2_len)
        jaccard_sim = same_word_len / float(len(set1 | set2))

        return same_word_len / float(max_len), same_word_len / float(min_len), same_word_len / float(avg_len), \
               unique_word1_len / float(word1_len), unique_word2_len /float(word2_len), jaccard_sim

    for index,row in train_data.iterrows():
        s1 = row['s1'].strip()
        s2 = row['s2'].strip()
        features = tuple()
        features = get_word_diff(s1,s2)
        for col_index,feature in enumerate(features):
            feature_train[index,col_index] = round(feature,5)

    for index, row in test_data.iterrows():
        s1 = row['s1'].strip()
        s2 = row['s2'].strip()
        features = tuple()
        features = get_word_diff(s1, s2)
        for col_index,feature in enumerate(features):
            feature_test[index,col_index] = round(feature,5)

    col_names = [('{}_{}'.format(feature_name,col_index))for col_index in range(col_num)]
    after_extract_feature_save_data(feature_train,feature_test,col_names,feature_name)


def extract_doubt_sim():
    '''
    抽取疑问词相同的比例
    '''
    # step1 定义抽取特征的方式名
    feature_name = 'nlp_doubt_sim'

    # step2 载入数据
    train_data, test_data = before_extract_feature_load_data(
        train_file=project.preprocessed_data_dir + 'train_0.6_seg.csv',
        test_file=project.preprocessed_data_dir + 'test_0.4_seg.csv')
    feature_train = np.zeros((train_data.shape[0], 1), dtype='float64')
    feature_test = np.zeros((test_data.shape[0],1),dtype='float64')

    doubt_words = load_doubt_words(project.aux_dir + 'doubt_words.txt')
    # 获取疑问词相同的比例
    def get_doubt_sim(q1, q2, doubt_words):
        q1_doubt_words = set(q1.split(" ")) & set(doubt_words)
        q2_doubt_words = set(q2.split(" ")) & set(doubt_words)
        return len(q1_doubt_words & q2_doubt_words) / float(len(q1_doubt_words | q2_doubt_words) + 1)
#delete for four ..decode('utf-8')
    for index,row in train_data.iterrows():
        # 因为doubt_words词表加载出来的是Unicode，所以需要将s1,s2解码成Unicode
        s1 = row['s1'].strip()
        s2 = row['s2'].strip()
        doubt_sim = get_doubt_sim(s1,s2,doubt_words)
        feature_train[index] = round(doubt_sim,5)

    for index, row in test_data.iterrows():
        s1 = row['s1'].strip()
        s2 = row['s2'].strip()
        doubt_sim = get_doubt_sim(s1, s2, doubt_words)
        feature_test[index] = round(doubt_sim,5)

    col_names = [feature_name]
    after_extract_feature_save_data(feature_train,feature_test,col_names,feature_name)


def extract_sentence_exist_topic():
    """
    抽取两个句子中是否同时存在蚂蚁花呗或者蚂蚁借呗的特征,同时包含花呗为1，同时包含借呗为1，否则为0
    :return:
    """
    # step1 定义抽取特征的方式名
    feature_name = 'nlp_sentece_exist_topic'

    # step2 载入数据
    train_data, test_data = before_extract_feature_load_data(
        train_file=project.data_dir + 'atec_nlp_sim_train_0.6.csv',
        test_file=project.data_dir + 'atec_nlp_sim_test_0.4.csv')
    feature_train = np.zeros((train_data.shape[0], 2), dtype='float64')
    feature_test = np.zeros((test_data.shape[0], 2), dtype='float64')

    def get_exist_same_topic(rawq1,rawq2):
        hua_flag = 0.
        jie_flag = 0.
        if '花呗' in rawq1 and '花呗' in rawq2:
            hua_flag = 1.

        if '借呗' in rawq1 and '借呗' in rawq2:
            jie_flag = 1.

        return hua_flag,jie_flag

    for index,row in train_data.iterrows():
        s1 = row['s1'].strip()
        s2 = row['s2'].strip()
        hua_flag, jie_flag = get_exist_same_topic(s1,s2)
        feature_train[index,0] = hua_flag
        feature_train[index,1] = jie_flag

    for index, row in test_data.iterrows():
        s1 = row['s1'].strip()
        s2 = row['s2'].strip()
        hua_flag, jie_flag = get_exist_same_topic(s1, s2)
        feature_test[index, 0] = hua_flag
        feature_test[index, 1] = jie_flag

    col_names = ['nlp_sentece_exist_topic_hua_flag','nlp_sentece_exist_topic_jie_flag']
    after_extract_feature_save_data(feature_train,feature_test,col_names,feature_name)


def extract_word_embedding_sim(w2v_model_path = 'train_all_data.bigram'):
    '''
    提取句子的词向量组合的相似度
    w2v_model_path为词向量文件
    :return:
    '''
    # step1 定义抽取特征的方式名
    feature_name = 'nlp_word_embedding_sim'

    # step2 载入数据
    train_data ,test_data = before_extract_feature_load_data(train_file=project.preprocessed_data_dir + 'train_0.6_seg.csv',
                                                             test_file=project.preprocessed_data_dir + 'test_0.4_seg.csv')
    feature_train = np.zeros((train_data.shape[0], 1), dtype='float64')
    feature_test = np.zeros((test_data.shape[0], 1), dtype='float64')

    train_all_w2v_model = KeyedVectors.load_word2vec_format(project.aux_dir + w2v_model_path, binary=False)

    # 得到句子的词向量组合（tfidf）
    def get_sen_vec(q, train_all_w2v_model, tfidf_dict, tfidf_flag=True):
        sen_vec = 0
        for word in q.split(' '):
            if word in train_all_w2v_model.vocab:
                word_vec = train_all_w2v_model.word_vec(word)
                word_tfidf = tfidf_dict.get(word, None)

                if tfidf_flag == True:
                    sen_vec += word_vec * word_tfidf
                else:
                    sen_vec += word_vec
        sen_vec = sen_vec / np.sqrt(np.sum(np.power(sen_vec, 2)) + 0.000001)
        return sen_vec

    def get_sentece_embedding_sim(q1, q2, train_all_w2v_model, tfidf_dict, tfidf_flag=True):
        # 得到两个问句的词向量组合
        q1_sec = get_sen_vec(q1, train_all_w2v_model, tfidf_dict, tfidf_flag)
        q2_sec = get_sen_vec(q2, train_all_w2v_model, tfidf_dict, tfidf_flag)

        # 曼哈顿距离
        # manhattan_distance = np.sum(np.abs(np.subtract(q1_sec, q2_sec)))

        # 欧式距离
        # enclidean_distance = np.sqrt(np.sum(np.power((q1_sec - q2_sec),2)))

        # 余弦相似度
        molecular = np.sum(np.multiply(q1_sec, q2_sec))
        denominator = np.sqrt(np.sum(np.power(q1_sec, 2))) * np.sqrt(np.sum(np.power(q2_sec, 2)))
        cos_sim = molecular / (denominator + 0.000001)

        # 闵可夫斯基距离
        # minkowski_distance = np.power(np.sum(np.power(np.abs(np.subtract(q1_sec, q2_sec)), 3)), 0.333333)
        # return manhattan_distance, enclidean_distance, cos_sim, minkowski_distance
        return cos_sim

    for index,row in train_data.iterrows():
        s1 = row['s1'].strip()
        s2 = row['s2'].strip()
        sentece_embedding_sim = get_sentece_embedding_sim(s1,s2,train_all_w2v_model,{},False)
        feature_train[index] = round(sentece_embedding_sim,5)

    for index, row in test_data.iterrows():
        s1 = row['s1'].strip()
        s2 = row['s2'].strip()
        sentece_embedding_sim = get_sentece_embedding_sim(s1, s2, train_all_w2v_model,{}, False)
        feature_test[index] = round(sentece_embedding_sim,5)

    col_names = [feature_name]
    after_extract_feature_save_data(feature_train,feature_test,col_names,feature_name)


if __name__ == '__main__':

    extract_sentence_length_diff()
    extract_edit_distance()
    extract_ngram()
    extract_sentence_diff_same()
    extract_doubt_sim()
    extract_sentence_exist_topic()
    extract_word_embedding_sim()

    pass