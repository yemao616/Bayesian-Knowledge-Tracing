"""
BKT

Run S3-PreprocessBKT and copy the data to the same folder as this file

"""
__author__ = 'adminuser'
from HMM import hmm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score, mean_squared_error

np.random.seed(1000)
folder = "/Users/Ye/Dropbox/LSTM/education/pyreness/bkt/"   # Specify your folder here
qlg_y = np.load(folder + "pyrenees_qlg.npy")                # target label for learning gain
post_y = np.load(folder + "pyrenees_post.npy")              # target value for post-test score


length = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # change length of sequences for early prediction
result = pd.DataFrame()
rmse, acc, recall, f1, auc, conf = [], [], [], [], [], []

for ind in length:
    print " "
    print "##############################################################################"
    print "########################################################### length of ", ind
    kf = KFold(n_splits=5, shuffle=True)
    qlg_train_pred, post_train_pred = [], []
    qlg_train_actual, post_train_actual = [], []
    qlg_test_pred, post_test_pred = [], []
    qlg_test_actual, post_test_actual = [], []

    numkc = 10      # change number of kc for different data set
    n = []

    for train_index, test_index in kf.split(qlg_y):
        qlg_y_train, qlg_y_test = qlg_y[train_index], qlg_y[test_index]
        post_y_train, post_y_test = post_y[train_index], post_y[test_index]

        # symbols here refers to four differnt observations:
        # fast-correct, fast-incorrect, slow-correct, slow-incorrect
        symbols = [['0', '1', '2', '3']]		
        h = hmm(2, Pi=np.array([0.8, 0.2]), T=np.array([[0.8, 0.2], [0.1, 0.9]]), obs_symbols=symbols)
        # change parameters here F= T=

        nlg_train = [[] for x in xrange(numkc)]
        nlg_test = [[] for x in xrange(numkc)]
        v = 0
        for i in xrange(numkc):
            X = np.load(folder + "new_bkt_pertime_kc" + str(i) + ".npy")
            tmp = []
            for each in X:
                end = int(len(each)*ind)        # for early prediction
                tmp.append(each[:end])

            X = np.array(tmp)
            X_train, X_test = X[train_index], X[test_index]

            train = [each for each in X_train if each]
            test = [each for each in X_test if each]
            if train and test:
                v += 1
                h.baum_welch(train, debug=False)
            nlg_train[i].extend(h.predict_nlg(X_train))
            nlg_test[i].extend(h.predict_nlg(X_test))

        n.append(v)
        print len(nlg_train), len(nlg_train[0])
        nlg_train = np.transpose(nlg_train)
        nlg_test = np.transpose(nlg_test)

        nlg_train = pd.DataFrame(nlg_train).fillna(value=0)
        nlg_test = pd.DataFrame(nlg_test).fillna(value=0)

        logreg = LogisticRegression()
        logreg.fit(nlg_train, pd.DataFrame(qlg_y_train))
        predict = logreg.predict(nlg_train)
        qlg_train_pred.extend([each for each in predict])
        qlg_train_actual.extend(qlg_y_train)

        predict = logreg.predict(nlg_test)
        print logreg.predict_proba(nlg_test)
        qlg_test_pred.extend([each for each in predict])
        qlg_test_actual.extend(qlg_y_test)

        # using MEAN for post prediction part #####
        # post_train = [each/numkc for each in np.sum(nlg_train, axis=1)]
        # post_test = [each/numkc for each in np.sum(nlg_test, axis=1)]
        #
        # post_train_pred.extend(post_train)
        # post_train_actual.extend(post_y_train)
        # post_test_pred.extend(post_test)
        # post_test_actual.extend(post_y_test)

        lg = LinearRegression()
        lg.fit(nlg_train, pd.DataFrame(post_y_train))
        predict = lg.predict(nlg_train)
        post_train_pred.extend([each for each in predict])
        post_train_actual.extend(post_y_train)

        predict = lg.predict(nlg_test)
        post_test_pred.extend([each for each in predict])
        post_test_actual.extend(post_y_test)

    # test code data ###
    # qlg_test_pred, post_test_pred = [0]*len(qlg_y), [0.5]*len(post_y)
    # qlg_test_actual, post_test_actual = qlg_y, post_y
    print " "
    print "<<<<<<< student learning gain"
    print "Training accuracy:" + str(accuracy_score(qlg_train_actual, qlg_train_pred))
    print "Accuracy: " + str(accuracy_score(qlg_test_actual, qlg_test_pred))
    acc.append(accuracy_score(qlg_test_actual, qlg_test_pred))


	# flip P and N here because we care about the low learning gain group: qlg = 0
    qlg_test_actual = [1 if each == 0 else 0 for each in qlg_test_actual]       
    qlg_test_pred = [1 if each == 0 else 0 for each in qlg_test_pred]


    print "f1_score: " + str(f1_score(qlg_test_actual, qlg_test_pred))
    f1.append(f1_score(qlg_test_actual, qlg_test_pred))
    print "Recall: " + str(recall_score(qlg_test_actual, qlg_test_pred))
    recall.append(recall_score(qlg_test_actual, qlg_test_pred))
    print "AUC: " + str(roc_auc_score(qlg_test_actual, qlg_test_pred))
    auc.append(roc_auc_score(qlg_test_actual, qlg_test_pred))
    print "Confusion Matrix: "
    print confusion_matrix(qlg_test_actual, qlg_test_pred)
    conf.append(confusion_matrix(qlg_test_actual, qlg_test_pred))
    print " "
    print "<<<<<<< student modeling"
    print "Training MSE: ", mean_squared_error(post_train_actual, post_train_pred)
    print "MSE: ", mean_squared_error(post_test_actual, post_test_pred)
    rmse.append(mean_squared_error(post_test_actual, post_test_pred)**0.5)
    print " trained kc: ", n


result['length'] = length
result['Accuracy'] = acc
result['Recall'] = recall
result['F1_measure'] = f1
result['AUC'] = auc
result['Confusion Matrix'] = conf
result['RMSE'] = rmse


result.to_csv('bkt_pyrenees.csv', index=False)         # change name for different data


y_set = [acc, recall, f1, auc]
for y in y_set:
    # plt.plot(length, y, 'bo', length, y, 'k')
    plt.plot(length, y, marker='o')
    for a, b in zip(length, y):
        plt.text(a, b, str(round(b, 3)))

plt.legend(['Accuracy', 'Recall', 'F1-measure', 'AUC'], loc=4)      # 1 means upper right, 4 means lower right
plt.show()