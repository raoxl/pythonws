# 特征选择
# 减少特征数量、降维，使模型泛化能力更强，减少过拟合,增强对特征和特征值之间的理解

# 1 去掉取值变化小的特征 Removing features with low variance

#2 单变量特征选择 Univariate feature selection
#        单变量特征选择能够对每一个特征进行测试，衡量该特征和响应变量之间的关系，根据得分扔掉不好的特征。对于回归和分类问题可以采用卡方检验等方式对特征进行测试。
#   2.1 Pearson相关系数 Pearson Correlation
#       Scikit-learn提供的 f_regrssion 方法能够批量计算特征的p-value，非常方便，参考sklearn的pipeline
import numpy as np
from scipy.stats import pearsonr
np.random.seed(0)
size = 300
x = np.random.normal(0, 1, size)
print("Lower noise", pearsonr(x, x + np.random.normal(0, 1, size)))
print("Higher noise", pearsonr(x, x + np.random.normal(0, 10, size)))

#   2.2 互信息和最大信息系数 Mutual information and maximal information coefficient (MIC)
#   2.3 距离相关系数 (Distance correlation)
#   2.4 基于学习模型的特征排序 (Model based ranking)
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

#Load boston housing dataset as an example
boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
for i in range(X.shape[1]):
     score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                              cv=ShuffleSplit(len(X), 3, .3))
     scores.append((round(np.mean(score), 3), names[i]))
print(sorted(scores, reverse=True))

#3 线性模型和正则化
#  回归模型，SVM，决策树，随机森林等等
from sklearn.linear_model import LinearRegression
import numpy as np
np.random.seed(0)
size = 5000
#A dataset with 3 features
X = np.random.normal(0, 1, (size, 3))
#Y = X0 + 2*X1 + noise
Y = X[:,0] + 2*X[:,1] + np.random.normal(0, 2, size)
lr = LinearRegression()
lr.fit(X, Y)
#A helper method for pretty-printing linear models
def pretty_print_linear(coefs, names = None, sort = False):
	if names == None:
		names = ["X%s" % x for x in range(len(coefs))]
	lst = zip(coefs, names)
	if sort:
		lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
	return " + ".join("%s * %s" % (round(coef, 3), name)
								   for coef, name in lst)
print("Linear model:", pretty_print_linear(lr.coef_))

#    3.1 正则化模型
#    3.2 L1正则化/Lasso
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

boston = load_boston()
scaler = StandardScaler()
X = scaler.fit_transform(boston["data"])
Y = boston["target"]
names = boston["feature_names"]

lasso = Lasso(alpha=.3)
lasso.fit(X, Y)

print "Lasso model: ", pretty_print_linear(lasso.coef_, names, sort = True)

#    3.3 L2正则化/Ridge regression


# 4 随机森林
#     4.1 平均不纯度减少 mean decrease impurity
#    4.2 平均精确率减少 Mean decrease accuracy

#5 两种顶层特征选择算法


#demo


from sklearn.datasets import load_boston
from sklearn.linear_model import (LinearRegression, Ridge,
								  Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from minepy import MINE
np.random.seed(0)
size = 750
X = np.random.uniform(0, 1, (size, 14))
#"Friedamn #1” regression problem
Y = (10 * np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2] - .5)**2 +
	 10*X[:,3] + 5*X[:,4] + np.random.normal(0,1))
#Add 3 additional correlated variables (correlated with X1-X3)
X[:,10:] = X[:,:4] + np.random.normal(0, .025, (size,4))
names = ["x%s" % i for i in range(1,15)]
ranks = {}
def rank_to_dict(ranks, names, order=1):
	minmax = MinMaxScaler()
	ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
	ranks = map(lambda x: round(x, 2), ranks)
	return dict(zip(names, ranks ))
lr = LinearRegression(normalize=True)
lr.fit(X, Y)
ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), names)
ridge = Ridge(alpha=7)
ridge.fit(X, Y)
ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)
lasso = Lasso(alpha=.05)
lasso.fit(X, Y)
ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)
rlasso = RandomizedLasso(alpha=0.04)
rlasso.fit(X, Y)
ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), names)
#stop the search when 5 features are left (they will get equal scores)
rfe = RFE(lr, n_features_to_select=5)
rfe.fit(X,Y)
ranks["RFE"] = rank_to_dict(map(float, rfe.ranking_), names, order=-1)
rf = RandomForestRegressor()
rf.fit(X,Y)
ranks["RF"] = rank_to_dict(rf.feature_importances_, names)
f, pval  = f_regression(X, Y, center=True)
ranks["Corr."] = rank_to_dict(f, names)
mine = MINE()
mic_scores = []
for i in range(X.shape[1]):
	mine.compute_score(X[:,i], Y)
	m = mine.mic()
	mic_scores.append(m)
ranks["MIC"] = rank_to_dict(mic_scores, names)
r = {}
for name in names:
	r[name] = round(np.mean([ranks[method][name]
							 for method in ranks.keys()]), 2)
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")
print ("\t%s" % "\t".join(methods))
for name in names:
	print ("%s\t%s" % (name, "\t".join(map(str,[ranks[method][name] for method in methods]))))



