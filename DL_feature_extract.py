# encoding = utf8
from DL_model import *
from Preprocessing import *
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils import to_categorical
from keras.models import load_model
import gc


# 深度学习特征提取
def extract_feature_siamese_lstm_manDist():
    # 前期参数设置
    embedding_matrix_file_path = 'train_all_w2v_embedding_matrix.pickle'
    feature_name = 'dl_siamese_lstm_manDist'
    RANOD_SEED = 42
    np.random.seed(RANOD_SEED)
    nepoch = 40
    num_folds = 5
    batch_size = 512

    # 加载Embeding矩阵
    embedding_matrix = project.load(project.aux_dir + embedding_matrix_file_path)

    #加载输入数据
    X_train_s1 = project.load(project.preprocessed_data_dir + 's1_train_ids_pad.pickle')
    X_train_s2 = project.load(project.preprocessed_data_dir + 's2_train_ids_pad.pickle')

    X_test_s1 = project.load(project.preprocessed_data_dir + 's1_test_ids_pad.pickle')
    X_test_s2 = project.load(project.preprocessed_data_dir + 's2_test_ids_pad.pickle')

    # y_0.6_train.pickle 存储的为list
    y_train = np.array(project.load(project.features_dir + 'y_0.6_train.pickle'))
    y_val = np.array(project.load(project.features_dir + 'y_0.4_test.pickle'))

    # 定义model param
    model_param = {
        'lstm_units':50,
        'lstm_dropout_rate':0.,
        'lstm_re_dropout_rate':0.,
        'desen_dropout_rate':0.75,
        'num_dense':128
    }
    # model_checkpoint_path = project.temp_dir + 'fold-checkpoint-'+feature_name + '.h5'
    kfold = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=RANOD_SEED
    )

    # 存放最后预测结果
    y_train_oofp = np.zeros((len(y_train),2),dtype='float64')
    y_test_oofp = np.zeros((len(X_test_s1),2),dtype='float64')

    train_y = to_categorical(y_train, 2)
    val_y = to_categorical(y_val,2)

    for fold_num, (ix_train, ix_val) in enumerate(kfold.split(X_train_s1,y_train)):

        # 选出需要添加的样本
        train_true_mask = y_train[ix_train] == 1
        X_train_true_s1 = X_train_s1[ix_train][train_true_mask]
        X_train_true_s2 = X_train_s2[ix_train][train_true_mask]
        y_train_true = train_y[ix_train][train_true_mask]

        # 进行添加
        X_add_train_fold_s1 = np.vstack([X_train_s1[ix_train],X_train_true_s2])
        X_add_train_fold_s2 = np.vstack([X_train_s2[ix_train],X_train_true_s1])
        y_add_train_fold = np.concatenate([train_y[ix_train],y_train_true])

        val_true_mask = y_train[ix_val] == 1
        X_val_true_s1 = X_train_s1[ix_val][val_true_mask]
        X_val_true_s2 = X_train_s2[ix_val][val_true_mask]
        y_val_true = train_y[ix_val][val_true_mask]

        # 进行添加
        X_add_val_fold_s1 = np.vstack([X_train_s1[ix_val], X_val_true_s2])
        X_add_val_fold_s2 = np.vstack([X_train_s2[ix_val], X_val_true_s1])
        y_add_val_fold = np.concatenate([train_y[ix_val], y_val_true])

        print ('start train fold {} of {} ......'.format((fold_num + 1), 5))
        # 创建模型
        model = create_siamese_lstm_ManDistance_model(embedding_matrix, model_param)
        # 训练模型
        model_checkpoint_path = project.trained_model_dir + 'dl_siamese_lstm_manDist_model{}.h5'.format(fold_num)
        model.fit(x=[X_add_train_fold_s1,X_add_train_fold_s2],y=y_add_train_fold,
                      validation_data=([X_add_val_fold_s1,X_add_val_fold_s2],y_add_val_fold),
                      batch_size=batch_size,
                      epochs=nepoch,
                      verbose=1,
                      class_weight={0: 1, 1: 2},
                      callbacks=[
                          EarlyStopping(
                              monitor='val_loss',
                              min_delta=0.005,
                              patience=5,
                              verbose=1,
                              mode='auto'
                          ),
                          ModelCheckpoint(
                              model_checkpoint_path,
                              monitor='val_loss',
                              save_best_only=True,
                              save_weights_only=False,
                              verbose=1
                          )]
                  )
        model.load_weights(model_checkpoint_path)
        y_train_oofp[ix_val] = predict(model,X_train_s1[ix_val],X_train_s2[ix_val])
        K.clear_session()
        del X_add_train_fold_s1
        del X_add_train_fold_s2
        del X_add_val_fold_s1
        del X_add_val_fold_s2
        del y_add_train_fold
        del y_add_val_fold
        gc.collect()

    # save feature

    model_path = project.trained_model_dir + 'dl_siamese_lstm_manDist_model0.h5'
    model0 = load_model(model_path,
                    custom_objects={'ManDist': ManDist, 'fbeta_score': fbeta_score, 'precision': precision,
                                        'recall': recall})
    y_test_oofp = predict(model0,X_test_s1,X_test_s2)
    col_names = ['{}_{}'.format(feature_name,index) for index in range(2)]
    after_extract_feature_save_data(y_train_oofp,y_test_oofp,col_names,feature_name)


def extract_feature_siamese_lstm_attention():
    # 前期参数设置
    embedding_matrix_file_path = 'train_all_w2v_embedding_matrix.pickle'
    feature_name = 'dl_siamese_lstm_attention'
    RANOD_SEED = 42
    np.random.seed(RANOD_SEED)
    nepoch = 50
    num_folds = 5
    batch_size = 512

    # 加载Embeding矩阵
    embedding_matrix = project.load(project.aux_dir + embedding_matrix_file_path)

    #加载输入数据
    X_train_s1 = project.load(project.preprocessed_data_dir + 's1_train_ids_pad.pickle')
    X_train_s2 = project.load(project.preprocessed_data_dir + 's2_train_ids_pad.pickle')

    X_test_s1 = project.load(project.preprocessed_data_dir + 's1_test_ids_pad.pickle')
    X_test_s2 = project.load(project.preprocessed_data_dir + 's2_test_ids_pad.pickle')

    #y_0.6_train.pickle 存储的为list
    y_train = np.array(project.load(project.features_dir + 'y_0.6_train.pickle'))
    y_val =  np.array(project.load(project.features_dir + 'y_0.4_test.pickle'))

    #定义model param
    model_param = {
        'lstm_units':50,
        'lstm_dropout_rate':0.,
        'lstm_re_dropout_rate':0.,
        'desen_dropout_rate':0.75,
        'num_dense':128
    }
    model_checkpoint_path = project.temp_dir + 'fold-checkpoint-'+feature_name + '.h5'
    kfold = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=RANOD_SEED
    )
    # 存放最后预测结果
    # y_train_oofp = np.zeros_like(y_train,dtype='float64')

    y_train_oofp = np.zeros((len(y_train),1),dtype='float64')

    y_test_oofp = np.zeros((len(X_test_s1),1),dtype='float64')

    for fold_num, (ix_train, ix_val) in enumerate(kfold.split(X_train_s1,y_train)):

        # 选出需要添加的样本
        train_true_mask = y_train[ix_train] == 1
        X_train_true_s1 = X_train_s1[ix_train][train_true_mask]
        X_train_true_s2 = X_train_s2[ix_train][train_true_mask]
        y_train_true = y_train[ix_train][train_true_mask]

        # 进行添加
        X_add_train_fold_s1 = np.vstack([X_train_s1[ix_train],X_train_true_s2])
        X_add_train_fold_s2 = np.vstack([X_train_s2[ix_train],X_train_true_s1])
        y_add_train_fold = np.concatenate([y_train[ix_train],y_train_true])



        val_true_mask = y_train[ix_val]==1
        X_val_true_s1 = X_train_s1[ix_val][val_true_mask]
        X_val_true_s2 = X_train_s2[ix_val][val_true_mask]
        y_val_true = y_train[ix_val][val_true_mask]

        # 进行添加
        X_add_val_fold_s1 = np.vstack([X_train_s1[ix_val], X_val_true_s2])
        X_add_val_fold_s2 = np.vstack([X_train_s2[ix_val], X_val_true_s1])
        y_add_val_fold = np.concatenate([y_train[ix_val], y_val_true])

        print ('start train fold {} of {} ......'.format((fold_num + 1), 5))
        # 创建模型
        model = create_siamese_lstm_attention_model(embedding_matrix, model_param)
        # 训练模型
        model_checkpoint_path = project.trained_model_dir + 'dl_siamese_lstm_attention_model{}.h5'.format(fold_num)
        model.fit(x=[X_add_train_fold_s1,X_add_train_fold_s2],y=y_add_train_fold,
                      validation_data=([X_add_val_fold_s1,X_add_val_fold_s2],y_add_val_fold),
                      batch_size=batch_size,
                      epochs=nepoch,
                      verbose=1,
                      class_weight={0: 1, 1: 2},
                      callbacks=[
                          EarlyStopping(
                              monitor='val_loss',
                              min_delta=0.005,
                              patience=5,
                              verbose=1,
                              mode='auto'
                          ),
                          ModelCheckpoint(
                              model_checkpoint_path,
                              monitor='val_loss',
                              save_best_only=True,
                              save_weights_only=False,
                              verbose=1
                          )]
                  )
        model.load_weights(model_checkpoint_path)
        y_train_oofp[ix_val] = predict(model,X_train_s1[ix_val],X_train_s2[ix_val])
        K.clear_session()
        del X_add_train_fold_s1
        del X_add_train_fold_s2
        del X_add_val_fold_s1
        del X_add_val_fold_s2
        del y_add_train_fold
        del y_add_val_fold
        gc.collect()

    model_path = project.trained_model_dir + 'dl_siamese_lstm_attention_model0.h5'
    model0 = load_model(model_path,
                        custom_objects={'AttentionLayer1': AttentionLayer1, 'fbeta_score': fbeta_score, 'precision': precision,
                                        'recall': recall})
    y_test_oofp = predict(model0, X_test_s1, X_test_s2)

    col_names = ['{}_{}'.format(feature_name,index) for index in range(1)]
    after_extract_feature_save_data(y_train_oofp,y_test_oofp,col_names,feature_name)


def extract_feature_siamese_lstm_dssm():
    # 前期参数设置
    embedding_matrix_file_path = 'train_all_w2v_embedding_matrix.pickle'
    embedding_char_matrix_file_path = 'train_all_char_embedding_matrix.pickle'
    feature_name = 'dl_siamese_lstm_dssm'
    RANOD_SEED = 42
    np.random.seed(RANOD_SEED)
    nepoch = 50
    num_folds = 5
    batch_size = 512

    # 加载Embeding矩阵
    embedding_matrix = project.load(project.aux_dir + embedding_matrix_file_path)
    char_embedding_matrix =  project.load(project.aux_dir + embedding_char_matrix_file_path)

    # 加载输入数据
    X_train_s1 = project.load(project.preprocessed_data_dir + 's1_train_ids_pad.pickle')
    X_train_s2 = project.load(project.preprocessed_data_dir + 's2_train_ids_pad.pickle')

    print(X_train_s2.shape)

    X_test_s1 = project.load(project.preprocessed_data_dir + 's1_test_ids_pad.pickle')
    X_test_s2 = project.load(project.preprocessed_data_dir + 's2_test_ids_pad.pickle')

    X_char_train_s1 = project.load(project.preprocessed_data_dir + 's1_train_char_ids_pad.pickle')
    X_char_train_s2 = project.load(project.preprocessed_data_dir + 's2_train_char_ids_pad.pickle')

    X_char_test_s1 = project.load(project.preprocessed_data_dir + 's1_test_char_ids_pad.pickle')
    X_char_test_s2 = project.load(project.preprocessed_data_dir + 's2_test_char_ids_pad.pickle')

    # y_0.6_train.pickle 存储的为list
    y_train = np.array(project.load(project.features_dir + 'y_0.6_train.pickle'))
    y_val = np.array(project.load(project.features_dir + 'y_0.4_test.pickle'))

    # train_y = to_categorical(y_train, 2)
    # val_y = to_categorical(y_val,2)
    # 定义model param
    model_param = {
        'lstm_units': 50,
        'lstm_dropout_rate': 0.,
        'lstm_re_dropout_rate': 0.,
        'desen_dropout_rate': 0.75,
        'num_dense': 128
    }
    kfold = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=RANOD_SEED
    )
    # 存放最后预测结果
    # y_train_oofp = np.zeros_like(y_train,dtype='float64')

    y_train_oofp = np.zeros((len(y_train), 1), dtype='float64')

    y_test_oofp = np.zeros((len(X_test_s1), 1), dtype='float64')

    for fold_num, (ix_train, ix_val) in enumerate(kfold.split(X_train_s1, y_train)):
        # 选出需要添加的样本
        train_true_mask = y_train[ix_train] == 1
        X_train_true_s1 = X_train_s1[ix_train][train_true_mask]
        X_train_true_s2 = X_train_s2[ix_train][train_true_mask]
        y_train_true = y_train[ix_train][train_true_mask]

        # 进行添加
        X_add_train_fold_s1 = np.vstack([X_train_s1[ix_train], X_train_true_s2])
        X_add_train_fold_s2 = np.vstack([X_train_s2[ix_train], X_train_true_s1])
        y_add_train_fold = np.concatenate([y_train[ix_train], y_train_true])



        X_train_true_s1_char = X_char_train_s1[ix_train][train_true_mask]
        X_train_true_s2_char = X_char_train_s2[ix_train][train_true_mask]

        # 进行添加
        X_add_train_fold_s1_char = np.vstack([X_char_train_s1[ix_train], X_train_true_s2_char])
        X_add_train_fold_s2_char = np.vstack([X_char_train_s2[ix_train], X_train_true_s1_char])

        #   验证部分
        val_true_mask = y_train[ix_val] == 1
        X_val_true_s1 = X_train_s1[ix_val][val_true_mask]
        X_val_true_s2 = X_train_s2[ix_val][val_true_mask]
        y_val_true = y_train[ix_val][val_true_mask]

        # 进行添加
        X_add_val_fold_s1 = np.vstack([X_train_s1[ix_val], X_val_true_s2])
        X_add_val_fold_s2 = np.vstack([X_train_s2[ix_val], X_val_true_s1])
        y_add_val_fold = np.concatenate([y_train[ix_val], y_val_true])

        X_val_true_s1_char = X_char_train_s1[ix_val][val_true_mask]
        X_val_true_s2_char = X_char_train_s2[ix_val][val_true_mask]

        X_add_val_fold_s1_char = np.vstack([X_char_train_s1[ix_val], X_val_true_s2_char])
        X_add_val_fold_s2_char = np.vstack([X_char_train_s2[ix_val], X_val_true_s1_char])

        print ('start train fold {} of {} ......'.format((fold_num + 1), 5))
        # 创建模型
        model = create_siamese_lstm_dssm_mdoel(embedding_matrix,char_embedding_matrix, model_param)
        # 训练模型
        model_checkpoint_path = project.trained_model_dir + 'dl_siamese_lstm_dssm_model{}.h5'.format(fold_num)
        model.fit(x=[X_add_train_fold_s1, X_add_train_fold_s2,X_add_train_fold_s1_char,X_add_train_fold_s2_char], y=y_add_train_fold,
                  validation_data=([X_add_val_fold_s1, X_add_val_fold_s2,X_add_val_fold_s1_char,X_add_val_fold_s2_char], y_add_val_fold),
                  batch_size=batch_size,
                  epochs=nepoch,
                  class_weight={0:1,1:2},
                  verbose=1,
                  callbacks=[
                      EarlyStopping(
                          monitor='val_loss',
                          min_delta=0.001,
                          patience=3,
                          verbose=1,
                          mode='auto'
                      ),
                      ModelCheckpoint(
                          model_checkpoint_path,
                          monitor='val_loss',
                          save_best_only=True,
                          save_weights_only=False,
                          verbose=1
                      )]
                  )
        model.load_weights(model_checkpoint_path)
        y_train_oofp[ix_val] = predict1(model, X_train_s1[ix_val], X_train_s2[ix_val],X_char_train_s1[ix_val],X_char_train_s2[ix_val])
        K.clear_session()
        del X_add_train_fold_s1
        del X_add_train_fold_s2
        del X_add_val_fold_s1
        del X_add_val_fold_s2
        del y_add_train_fold
        del y_add_val_fold
        gc.collect()

    model_path = project.trained_model_dir + 'dl_siamese_lstm_dssm_model0.h5'
    model0 = load_model(model_path,
                        custom_objects={'AttentionLayer': AttentionLayer,'ManDist': ManDist,'ConsDist':ConsDist, 'fbeta_score': fbeta_score,
                                        'precision': precision,
                                        'recall': recall})
    y_test_oofp = predict1(model0, X_test_s1, X_test_s2,X_char_test_s1,X_char_test_s2)
    col_names = ['{}_{}'.format(feature_name, index) for index in range(1)]
    after_extract_feature_save_data(y_train_oofp, y_test_oofp, col_names, feature_name)


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


if __name__ == '__main__':
    extract_feature_siamese_lstm_manDist()
    extract_feature_siamese_lstm_attention()
    extract_feature_siamese_lstm_dssm()
    pass
