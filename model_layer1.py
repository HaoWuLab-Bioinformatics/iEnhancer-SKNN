import numpy as np
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score

x_train = np.loadtxt('./Features/enhancer_identification/features_layer1_train.txt')
y_train = np.loadtxt('./Features/enhancer_identification/lable_layer1_train.txt')
x_test = np.loadtxt('./Features/enhancer_identification/features_layer1_test.txt')
y_test = np.loadtxt('./Features/enhancer_identification/lable_layer1_test.txt')

base_model1 = KNeighborsClassifier(leaf_size=1, n_neighbors=17, p=1, weights='distance')
base_model2 = KNeighborsClassifier(leaf_size=1, n_neighbors=18, p=1, weights='distance')
base_model3 = KNeighborsClassifier(leaf_size=1, n_neighbors=19, p=1, weights='distance')
base_model4 = KNeighborsClassifier(leaf_size=1, n_neighbors=20, p=1, weights='distance')
base_model5 = KNeighborsClassifier(leaf_size=1, n_neighbors=23, p=1, weights='distance')
meta_model = LogisticRegression(random_state=10, max_iter=50000, penalty='l2', C=1.0, multi_class='multinomial', solver='sag')
# enhancer identification: model11, model12, model13, model14, model15
# enhancer strength classification: model21, model22, model23, model24, model25
stack = StackingCVClassifier(
   classifiers=[base_model1, base_model2, base_model3, base_model4, base_model5],
    meta_classifier=meta_model, random_state=10, use_probas=True, cv=5)
n_folds = 10
kf = KFold(10, True, 10)
i = 0
acc_stack = np.zeros(10)
mcc = np.zeros(10)
SN = np.zeros(10)
for train_index, test_index in kf.split(x_train):
    stack.fit(x_train[train_index], y_train[train_index])
    stack_pred = stack.predict_proba(x_train[test_index])
    stack_predict = stack_pred[:, 1]
    stack_p = stack.predict(x_train[test_index])
    acc_stack[i] = accuracy_score(y_train[test_index], stack_p)
    fpr, tpr, thresholdTest = roc_curve(y_train[test_index], stack_predict)
    SN[i] = recall_score(y_train[test_index], stack_p)
    mcc[i] = matthews_corrcoef(y_train[test_index], stack_p)
    i = i + 1

stack.fit(x_train, y_train)
stack_pred = stack.predict_proba(x_test)
stack_predict = stack_pred[:, 1]
stack_p = stack.predict(x_test)
acc_stack_test = accuracy_score(y_test, stack_p)
fpr, tpr, thresholdTest = roc_curve(y_test, stack_predict)
mcc_test = matthews_corrcoef(y_test, stack_p)
sn_test = recall_score(y_test, stack_p)
print('The performance of enhancer identification：')
print('acc_test:', acc_stack_test)
print('sn_test:', sn_test)
print('mcc_test:', mcc_test)



