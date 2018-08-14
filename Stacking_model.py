#For the ensemble learning
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np

class StackingBaseClassifier(object):

    def train(self, x_train, y_train, x_val=None, y_val=None):
        pass

    def predict(self, model, x_test):
        pass

    def get_model_out(self, x_train, y_train, x_test, n_fold=5):
        n_train = x_train.shape[0]
        n_test = x_test.shape[0]
        train_oofp = np.zeros((n_train,))  # 存储每个fold的预测结果
        test_oofp = np.zeros((n_test, n_fold))  # 存储对测试集预测结果

        kfold = KFold(n_splits=n_fold, random_state=44, shuffle=True)


        for index, (ix_train, ix_val) in enumerate(kfold.split(x_train)):
            print (('{} fold of {} start train and predict...').format(index, n_fold))
            X_fold_train = x_train[ix_train]
            y_fold_train = y_train[ix_train]

            X_fold_val = x_train[ix_val]
            y_fold_val = y_train[ix_val]

            model = self.train(X_fold_train, y_fold_train, X_fold_val, y_fold_val)

            train_oofp[ix_val] = self.predict(model, X_fold_val)
            test_oofp[:, index] = self.predict(model, x_test)

        test_oofp_mean = np.mean(test_oofp, axis=1)
        return train_oofp, test_oofp_mean


class GaussianNBClassifier(StackingBaseClassifier):
    def __init__(self):
        # 参数设置
        pass

    def train(self, x_train, y_train, x_val, y_val):
        print ('use GaussianNB train model...')
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)
        return gnb #, gnb.score(x_val, y_val)

    def predict(self, model, x_test):
        print ('use GaussianNB model test... ')
        return model.predict(x_test)


class LGBClassifier(StackingBaseClassifier):
    def __init__(self):
        self.lgb_param = {
            'objective': 'binary',
            'metric': {'auc', 'binary_logloss'},
            'boosting': 'gbdt',
            'device': 'cpu',
            'feature_fraction': 0.8,  # 抽取所有特征的0.75个进行训练
            'num_leaves': 16,
            'learning_rate': 0.01,
            'verbose': 1,
            'bagging_seed': 456,
            'feature_fraction_seed': 456
        }

    def train(self, x_train, y_train, x_val, y_val):
        print ('use LGB train model...')
        lgb_data_train = lgb.Dataset(x_train, y_train)
        lgb_data_val = lgb.Dataset(x_val, y_val)
        evals_res = {}

        model = lgb.train(
            params=self.lgb_param,
            train_set=lgb_data_train,
            valid_sets=[lgb_data_train, lgb_data_val],  # 训练集和测试集都需要验证
            valid_names=['train', 'val'],
            evals_result=evals_res,
            num_boost_round=2500,
            early_stopping_rounds=10,
            verbose_eval=False
        )
        return model

    def predict(self, model, x_test):
        print('use LGB model test...')
        return model.predict(x_test)


class RFClassifer(StackingBaseClassifier):
    def train(self, x_train, y_train, x_val, y_val):
        print('use RandomForest train model...')
        clf = RandomForestClassifier(n_estimators=25,
                                     max_depth=4,
                                     class_weight={
                                         0: 1,
                                         1: 4
                                     }
                                     )
        clf.fit(x_train, y_train)
        return clf #, 0.

    def predict(self, model, x_test):
        print ('use RandomForest test...')
        return model.predict(x_test)


class LogisicClassifier(StackingBaseClassifier):
    def train(self, x_train, y_train, x_val=None, y_val=None):
        print ('use LogisticRegression train model...')
        lr = LogisticRegression(class_weight={0: 1, 1: 4})
        lr.fit(x_train, y_train)
        return lr
    def predict(self, model, x_test):
        print ('use LogisticRegression test...')
        return model.predict(x_test)


class DecisionClassifier(StackingBaseClassifier):
    def train(self, x_train, y_train, x_val=None, y_val=None):
        print('use DecisionClassifier train model...')
        dt = DecisionTreeClassifier(class_weight={0: 1, 1: 4},max_depth=5)
        dt.fit(x_train, y_train)
        return dt

    def predict(self, model, x_test):
        print ('use DecisionClassifier test...')
        return model.predict(x_test)


class SVMClassifier(StackingBaseClassifier):
    def train(self, x_train, y_train, x_val=None, y_val=None):
        print('use SVMClassifier train model...')
        SVMC = SVC(C=1.0, kernel='rbf', degree=3,
                   shrinking=True, probability=False, tol=0.001, cache_size=200,
                   class_weight={0: 1, 1: 4}, max_iter=1000,
                   decision_function_shape='ovr', random_state=None)
        tmp = MinMaxScaler()
        tmp.fit(x_train)
        SVMC.fit(x_train,y_train)
        return SVMC

    def predict(self, model, x_test):
        print('use SVMClassifier test...')
        return model.predict(x_test)