# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def sigmoid(h):
#     return 1.0 / (1.0 + np.exp(-h))
#
#
# h = np.arange(-10, 10, 0.1)
# s_h = sigmoid(h)
# plt.plot(h, s_h)
# plt.axvline(0.0, color='k')
# plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
# plt.axhline(y=0.5, ls='dotted', color='k')
# plt.yticks([0.5, 0.5, 1.0])
# plt.ylim(-0.1, 1.1)
# plt.xlabel('h')
# plt.ylabel('$S(h)$')
# plt.title('sigmoid func')
# plt.show()



import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from  sklearn import datasets
from  sklearn.cross_validation import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from  sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
sc.mean_
sc.scale_
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
# n_iter：可以理解成梯度下降中迭代的次数
# eta0：可以理解成梯度下降中的学习率
# random_state：设置随机种子的，为了每次迭代都有相同的训练集顺序

ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print( accuracy_score(y_test, y_pred))

#序列化
import pickle
# f=open('Perceptron_Model.txt', 'wb')
# pickle.dump(ppn,f)
# f.close()

#反序列化
# f = open('Perceptron_Model.txt', 'rb')
# d = pickle.load(f)
# f.close()
# y_pred1 = d.predict( X_test_std )
# print(accuracy_score(y_test, y_pred1))



X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(C=1000.0,random_state=0)
lr.fit(X_train_std, y_train)
print(lr.predict_proba(X_test_std[0,:]))

from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_combined_std, y_combined,lr)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')





#=====================================================================================================
#第三章

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc",size=10)

def runplt():
    plt.figure()
    plt.title('匹萨价格与直径数据',fontproperties=font)
    plt.xlabel('直径（英寸）', fontproperties=font)
    plt.ylabel('价格（美元）', fontproperties=font)
    plt.axis([0, 25, 0, 25])
    plt.grid(True)
    return plt

plt=runplt()
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]
plt.plot(X,y,'k.')
plt.show()













