
# coding: utf-8

# numpy

# In[464]:

import numpy  as np
import pandas as pd
import seaborn as sns
import time,timeit
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
adress='/opt/workspace/ipython_ws/xlrao/bigtable.txt'
pd=pd.read_csv(adress,sep='\t',low_memory=False)
pd.describe()
#reader = pd.read_csv(adress, sep = '\t', dtype = {'uid': str,'gold_user_level':str}, iterator = True)
pd.ix[1:40,0:10]
pd.shape

#python数据挖掘领域工具包  http://www.jianshu.com/p/9f76fc8fec49


# In[524]:

#esc h 查看快捷键
#'data'%mkdir$folder  #当前路径下建新文件夹
#在py变量前面加入$, 可以把这个变量共享给OS或者magic command:


#.%bookmark
#与系统交互
#在IPython里调用系统的命令, 不用再使用sys.exec('...')之类冗长的方式了, 只需要在系统的命令前面加上一个感叹号!即可...
file = get_ipython().getoutput('ls -l -S # | grep edges')
file
get_ipython().system('pwd')


# In[522]:

#魔术命令

get_ipython().magic('reset ##删除interactive命名空间中的全部变量 /名称')

get_ipython().magic('xdel variable  # 删除variable，并尝试清楚骑在Ipython中的对象上的一切引用')

get_ipython().magic('hist #用于打印全部或部分输入历史')


# In[ ]:

get_ipython().magic('logstart #开始记录日志    # %logoff  ,%logon  %logstate   %logstop')


# In[ ]:

get_ipython().magic('run ipython_script_test.py    #在I Python会话环境中，所有文件都可以通过%run命令当做Python程序来运行。')


# In[380]:

get_ipython().magic('quickref #')
get_ipython().magic('magic')


# In[529]:

data1=[]
type(data1)



# In[492]:

# collections是Python内建的一个集合模块，提供了许多有用的集合类。
# http://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/001411031239400f7181f65f33a4623bc42276a605debf6000
import collections
collections.__all__


# In[500]:

#时间日期
from datetime import datetime

now = datetime.now() # 获取当前datetime
print(now)

dt = datetime(2015, 4, 19, 12, 20) # 用指定日期时间创建datetime
print(dt)

dt.timestamp() # 把datetime转换为timestamp  注意Python的timestamp是一个浮点数。如果有小数位，小数位表示毫秒数。

t = 1429417200.0
print(datetime.fromtimestamp(t))

cday = datetime.strptime('2015-6-1 18:19:59', '%Y-%m-%d %H:%M:%S')
print(cday)

now = datetime.now()
print(now.strftime('%a, %b %d %H:%M'))

from datetime import datetime, timedelta
now = datetime.now()
now + timedelta(hours=10)
now - timedelta(days=1)
now + timedelta(days=2, hours=12)






# In[507]:



import os
os.name
os.uname

os.environ           #环境变量  os.environ.get('key')

# 查看当前目录的绝对路径:
>>> os.path.abspath('.')
#'/Users/michael'
# 在某个目录下创建一个新目录，首先把新目录的完整路径表示出来:
>>> os.path.join('/Users/michael', 'testdir')
#'/Users/michael/testdir'
# 然后创建一个目录:
>>> os.mkdir('/Users/michael/testdir')
# 删掉一个目录:
>>> os.rmdir('/Users/michael/testdir')


同样的道理，要拆分路径时，也不要直接去拆字符串，而要通过os.path.split()函数，这样可以把一个路径拆分为两部分，后一部分总是最后级别的目录或文件名：

>>> os.path.split('/Users/michael/testdir/file.txt')
('/Users/michael/testdir', 'file.txt')
os.path.splitext()可以直接让你得到文件扩展名，很多时候非常方便：

>>> os.path.splitext('/path/to/file.txt')
('/path/to/file', '.txt')
这些合并、拆分路径的函数并不要求目录和文件要真实存在，它们只对字符串进行操作。

文件操作使用下面的函数。假定当前目录下有一个test.txt文件：

# 对文件重命名:
>>> os.rename('test.txt', 'test.py')
# 删掉文件:
>>> os.remove('test.py')





# In[516]:

# 序列化
d = dict(name='Bob', age=20, score=88)
import pickle
pickle.dumps(d)   #pickle.dumps()方法把任意对象序列化成一个bytes，然后，就可以把这个bytes写入文件。或者用另一个方法pickle.dump()直接把对象序列化后写入一个file-like Object：

f=open('dump1.txt', 'wb')
pickle.dump(d, f)
f.close()


f = open('dump1.txt', 'rb')
d = pickle.load(f)
f.close()
d


#jason
# import json
# d = dict(name='Bob', age=20, score=88)
# json.dumps(d)
# '{"age": 20, "score": 88, "name": "Bob"}'


# In[520]:

# 进程

import os

print('Process (%s) start...' % os.getpid())
# Only works on Unix/Linux/Mac:
pid = os.fork()
if pid == 0:
    print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
else:
    print('I (%s) just created a child process (%s).' % (os.getpid(), pid))



    
from multiprocessing import Process
import os

# 子进程要执行的代码
def run_proc(name):
    print('Run child process %s (%s)...' % (name, os.getpid()))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Process(target=run_proc, args=('test',))
    print('Child process will start.')
    p.start()
    p.join()
    print('Child process end.')
    
    
    


# In[378]:

#输入和输出变量
#忘记把函数结果赋值给变量是一件让人很郁闷的事情。好在iPython会将输入（你输入的那些文本）和输出（返回的对象）的引用保存在一些特殊变量中。
#最近的两个输出结果分别保存在－ （一个下划线）和－－（两个下划线）变量中：

_
__
_2   #（输出变量）
_i2  # (输入变量）



get_ipython().magic('env')


# In[ ]:




# In[221]:

get_ipython().magic('history # 或者%hist, 显示之前的记录, 有一些参数可用...')
get_ipython().magic('store # 把python变量的内容保存下来, 以后的session可以用')
get_ipython().magic('paste # 导入并执行剪贴板里面的内容')
get_ipython().magic('run # 之前讲过了, 运行py文件, 运行后py文件里的变量可以在console里访问')
get_ipython().magic('edit # 打开系统的文件编辑器, 并且在关闭这个编辑器时自动运行程序')


# In[474]:

get_ipython().magic('pwd')
get_ipython().system('pwd')

#matplotlib是最著名的Python图表绘制扩展库，它支持输出多种格式的图形图像，并且可以使用多种GUI界面库交互式地显示图表
#使用%matplotlib命令可以将matplotlib的图表直接嵌入到Notebook之中，或者使用指定的界面库显示图表，它有一个参数指定matplotlib图表的显示方式。inline表示将图表嵌入到Notebook中。

#IPython提供了许多魔法命令，使得在IPython环境中的操作更加得心应手。
#魔法命令都以%或者%%开头，以%开头的成为行命令，%%开头的称为单元命令
#行命令只对命令所在的行有效，而单元命令则必须出现在单元的第一行，对整个单元的代码进行处理。


# In[477]:

get_ipython().magic('timeit [x*x for x in range(1000)]')


# In[478]:

import collections  as col
import json               
#?显示文件
#??显示源代码


# In[538]:

print(np.arange(10) )

print(range(10))


import datetime as dt

dt.datetime.now()



# In[ ]:

#NumPy 数组和矢量计算 （Numerical Python）
#ndarry,矢量运算及复杂广播能力的多位数组

# 1、数据整理，清理，子集构造和过滤、转换等快速的矢量化数组运算
# 2、常用数组算法，如排序，唯一化，集合运算
# 3、高效的描述统计和数据聚合，摘要运算
# 4、异构数据的合并连

# 5、将条件逻辑表述为数组表达式
# 6、数据分组运算（聚合，转换，函数）


# In[223]:

#ndarry N维数组对象  ,通用同构数据容器   shape 表示各维大小，dtype 说明数据类型
data1=[2,4,6,8,9,6,4,3,2,5]
arr1=np.array(data1,dtype=np.float64)
type(arr1)  #numpy.ndarray
type(data1) #list
arr1=arr1.reshape(5,2)   
arr1.dtype
arr1.shape
arr1.ndim    #计算维度数
arr1.all()
np.zeros(10)   #array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])   创建数组值为0的数组
np.empty((2,3,2)) #返回值为随机数（垃圾值）
np.ones(10)       #array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
np.arange(15)     #与python 内置range()相似
#np.ones_like(arr1, dtype=None, order='K', subok=True)    # np.ones_like? , np.ones_like??  查看方法具体定义及参数
np.eye(5)              #生成方正，对角线为1
np.identity(5)         #生成方正，对角线为1
float_arr=arr1.astype(np.float64)   #转换类型       np.string_  ，np.unicode_



# In[224]:

#数组与标量之间的运算   广播(broadcasting)，元素及
arr1*2
arr1*arr1
arr1-arr1


# In[225]:

#基本索引及切片
arr2=np.arange(10)
arr2[5:7]=12     
arr3=arr2[:].copy()   # 如果需要得到切片而非视图则需要显示进行复制  .copy
arr1[1,1]  #arr1[1][1]  这两种方法等价    二维先行在列，一维 列


# In[226]:

arr1[:]


# In[228]:

arr1[2:4][0]


# In[ ]:




# In[ ]:




# In[231]:

#布尔索引
from numpy import random
names=np.array(['bob','joe','will','bob','will','joe','joe'])
data=random.randn(7,4)
names=='bob'
data[names=='bob',2:]

#data[-(names=='bob'),2:]
data[names!='bob',2:]
data[(names=='bob')|(names=='joe')]   #不可以换成 python的  and  or 
data[data<0]=0


# In[383]:




# In[232]:

#花式索引   ,会复制数据
data[np.ix_([1,3],[2,3])]


# In[233]:

#利用数组进行数据处理   ，用数组表达式代替循环的做法，通常被称为矢量化   ，通常矢量化运算要比等价纯python快一两个数量级
import matplotlib.pyplot as plt
points=np.arange(-5,5,0.01)  
xs,ys = np.meshgrid(points,points)
z = np.sqrt(xs**2 + ys**2)
z


# In[504]:

get_ipython().magic('matplotlib inline')
plt.imshow(z,cmap=matplotlib.pyplot.cm.Blues)
plt.colorbar()


# In[236]:

get_ipython().magic('pylab inline')
#%pylab qt
x=linspace(-2,2,10)
plot(x,sin(x))


# In[238]:

import os
os.getcwd()


# In[242]:

#将条件逻辑表述为数组运算
xarr= np.array([1.1,1.2,1.3,1.4,1.5])
yarr= np.array([2.1,2.2,2.3,2.4,2.5])
cond= np.array([True,False,True,True,False])

result =[(x if c else y ) for x,y,c in zip(xarr,yarr,cond)]
result

result1=np.where(cond,xarr,yarr)
result1


# In[508]:

arr3=random.randn(4,4)
np.where(arr3>0,2,-2)


# In[38]:

#数学和统计方法
arr4=random.randn(5,4)
arr4.mean()
arr4.sum()   

arr4.mean(axis=1)   #结果是少一维的数组      sumsum,cumrod 
(arr4>0).mean()
np.any(arr4>0)
np.all(arr4>0)
arr4=random.randn(5,4)


# In[386]:

#排序
arr5=random.randn(8)
arr5.sort()
arr5

arr5=random.randn(5,4)
arr5.sort(0)      # 0 纵向,1 横向轴编号
np.unique(arr5)   #唯一化


# In[398]:

b = np.linspace(0,pi,10)     #产生指定长度的数组    (start  ,end  , num)
arange( 10, 30, 5 )          #步长


# In[400]:

for a in b.flat:   #然而，如果一个人想对每个数组中元素进行运算，我们可以使用flat属性，该属性是数组元素的一个迭代器:
    print(a)
    
list(b.flat)  


#http://blog.csdn.net/huahaitingyuan/article/details/41419937


# In[405]:

a = floor(10*random.random((3,4)))
a.shape                             # reshape函数改变参数形状并返回它，而resize函数改变数组自身。


a.ravel() # flatten the array       散开
#array([ 7.,  5.,  9.,  3.,  7.,  2.,  7.,  8.,  6.,  8.,  3.,  2.])
a.shape = (6, 2)
a.transpose()
a


# In[410]:

#组合(stack)不同的数组    vstack   hstack

#几种方法可以沿不同轴将数组堆叠在一起：

a = floor(10*random.random((2,2)))
print(a)
b = floor(10*random.random((2,2)))
print(b)
print(vstack((a,b)))
print(hstack((a,b)))

column_stack((a,b)) 
#row_stack函数，另一方面，将一维数组以行组合成二维数组。

#对那些维度比二维更高的数组，hstack沿着第二个轴组合，vstack沿着第一个轴组合,concatenate允许可选参数给出组合时沿着的轴。


# In[412]:

#将一个数组分割(split)成几个小数组    vsplit沿着纵向的轴分割，array split允许指定沿哪个轴分割。

a = floor(10*random.random((2,12)))

print(a)
hsplit(a,3)   # Split a into 3        ,vsplit


# In[ ]:

#视图(view)和浅复制

c = a.view()
c is a
c.base is a                        # c is a view of the data owned by a
c.flags.owndata
c.shape = 2,6                      # a's shape doesn't change
a.shape
c[0,4] = 1234                      # a's data changes
a



# In[ ]:

#函数和方法(method)总览
这是个NumPy函数和方法分类排列目录。这些名字链接到NumPy示例,你可以看到这些函数起作用。5
创建数组
arange, array, copy, empty, empty_like, eye, fromfile, fromfunction, identity, linspace, logspace, mgrid, ogrid, ones, ones_like, r , zeros, zeros_like 
转化
astype, atleast 1d, atleast 2d, atleast 3d, mat 
操作
array split, column stack, concatenate, diagonal, dsplit, dstack, hsplit, hstack, item, newaxis, ravel, repeat, reshape, resize, squeeze, swapaxes, take, transpose, vsplit, vstack 
询问
all, any, nonzero, where 
排序
argmax, argmin, argsort, max, min, ptp, searchsorted, sort 
运算
choose, compress, cumprod, cumsum, inner, fill, imag, prod, put, putmask, real, sum 
基本统计
cov, mean, std, var 
基本线性代数
cross, dot, outer, svd, vdot


# In[419]:

#take
a[1:,].take([1,3],axis=1)
a
a[ix_((0,1),(1,5))]   #还有一种方法是通过矩阵向量积(叉积)。


# In[420]:

#直方图(histogram)

#NumPy中histogram函数应用到一个数组返回一对变量：直方图数组和箱式向量。注意：matplotlib也有一个用来建立直方图的函数(叫作hist,正如matlab中一样)与NumPy中的不同。
#主要的差别是pylab.hist自动绘制直方图，而numpy.histogram仅仅产生数据。

import numpy
import pylab
# Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2
mu, sigma = 2, 0.5
v = numpy.random.normal(mu,sigma,10000)
# Plot a normalized histogram with 50 bins
pylab.hist(v, bins=50, normed=1)       # matplotlib version (plot)
pylab.show()
# Compute the histogram with numpy and then plot it
(n, bins) = numpy.histogram(v, bins=50, normed=True)  # NumPy version (no plot)
pylab.plot(.5*(bins[1:]+bins[:-1]), n)
pylab.show()


# In[464]:

print(bins.shape)
print(n.shape)

pylab.plot(bins[1:],n)
#pylab.show()


# In[437]:

print(bins)
print(bins.shape)
print(bins[:-1])
print(bins[1:])

bins[1:]+bins[:-1]
n


# In[58]:

#用于数组的文件输入输出, 保存数据

np.save('arr_savetest',arr5)     # arr_savetest.npy
np.savez('arr_zsavetest',a=arr5,b=arr5)    #arr_zsavetest.npz   压缩
np.load('arr_savetest.npy') 

#存取文本文件
#pd.read_csv()
#pd.read_table()
load('arr_zsavetest').file
np.savetxt('testtxt',arr5)
np.loadtxt('testtxt')   #,delimiter=','
get_ipython().system('cat testtxt')
#np.genformtxt（）      #将数据加载到普通的NumPy   只不过它面向的是结构化数组和缺失数据处理


# In[65]:

#线性代数
x = np.array([[1,2.,3.], [4., 5. , 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
np.dot(x,y)    #dot  矩阵乘法

np.dot(x,np.ones(3))
#np.linalg 中有一组标准的矩阵分解运算以及诸如求逆和行列式之类的东西
from numpy.linalg import inv,qr   计算QR分解

diag  以一维数组的形式返回方阵的对角线（或非对角线）元素，或将一维数组转换为方阵（非对角线元素为o)
dot   矩阵乘法
trace 计算对角线元素的和
det   计算矩阵行列式
eig   计算方阵的本征值和本征向量
1nv   计算方阵的逆
p1nv  计算矩阵的Moore - Penrose伪逆
qr    计算QR分解
svd   计算奇异值分解（ SVD)
solve 解线性方程组Ax= b ，其中A 为一个方阵
lstsq 计算Ax= b的最小二乘解


# In[384]:

import numpy.linalg 


# In[83]:

#生成随机数
#from numpy.random import normalvariate
seed         确定随机数生成器的种子
permutation  返回一个序列的随机排列或返回一个随机排列的范围
shuffle      对一个序列就地随机排列
rand         产生均匀分布的样本值
randint      从给定的上下限范围内随机选取整数
randn        产生正态分布（平均值为0 ，标准差为1 ）的样本值，类似于MATLAB接口
binomial     产生二项分布的样本值
normal       产生正态（高斯）分布的样本值
beta         产生Beta分布的样本值
函鼓
chisquare    产生卡方分布的样本值
gamma        产生Gamma分布的样本值
uniform      产生在［O, 1 ）中均匀分布的样本值

get_ipython().magic('timeit samples = [random.normal(0,1) for i in  range(100)]    #xrange 3.0版本被去除')


# In[510]:

position =0
walk=[position]
steps=1000
for i in range(steps):
    # step = 1 if random.randint(0,1) else -1
    step = np.where(random.normal(0,1) >0,1,-1)
    position +=step
    walk.append(position)

plot(walk)  


# In[ ]:




# In[135]:

nsteps=1000
draws=np.random.randint(0,2,size=nsteps)
steps = np.where(draws>0,1,-1)
walk = steps.cumsum()
walk.max()
walk.min()
plot(walk)

(np.abs(walk)>10).argmax()
(np.abs(walk)>10).max()
#首先min/max与np.argmin/np.argmax函数的功能不同 前者返回值，后者返回最值所在的索引（下标）
#处理的对象不同 前者跟适合处理list等可迭代对象，而后者自然是numpy里的核心数据结构ndarray（多维数组）
#min/max是python内置的函数 np.argmin/np.argmax是numpy库中的成员函数


# In[143]:

nwalks=5000
nsteps=1000
draws=np.random.randint(0,2,size=(nwalks,nsteps))
steps=np.where(draws>0,1,-1)
walks=steps.cumsum(1)     # axis=1  行， 0 列
walks
hits30 =(np.abs(walks)>= 30).any(1)
hits30.sum()
crossing_times=(np.abs(walks[hits30])>= 30).argmax(1)
crossing_times.mean()


# pandas

# In[465]:

from pandas import Series, DataFrame   #(Series, DataFrame 使用非常多，直接引入本地命名空间)
import pandas as pd

#Series   Series是一种类似干一维数组的对象，它由一组数据（各种NumPy数据类型）以及一组与之相关的数据标签（即索引｜）组成。仅由一组数据即可产生最简单的Series : 
obj=Series([4,7,-5,3])
obj.values
obj.index

obj2 = Series([ 4, 7, -5, 3],index=['d','b','a','c'])    #指定索引  ， 如果不指定会自动生成
obj2['a']=9     #根据索引修改值

obj2[obj2>5]   #NumPy数组运算（如根据布尔型数组进行过捷、标量乘法、应用数学函数等）都会保留索引和值之间的链接
obj2*2
np.exp(obj2)
'b' in obj2    #还可以将Series看成是一个定长的有序字典，因为它是索引值到数据值的一个映射。它可以用在许多原本需要字典参数的函数中：

sdata = {'Ohio':35000, 'Texas':7100,'Oregon':16000,'tah':5000}
obj3=Series(sdata)    #通过字典创建 Series

pd.isnull(obj3)  # pd.notnull(obj3)   #pandas的isnull罪IJnotnull 函数可用于检测缺失数据：
#对于许多应用而言， Series最重要的一个功能是：它在算术运算中会自动对齐不同索引的数据。


#Series对象本身及其索引都有一个name属性，该属性跟pandas其他的关键功能关系非常密切：

obj3.name='population'
obj3.index.name='state'
obj3




# In[637]:

#DataFrame    是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符E辑、布尔1在等） D ataF ram e既有行索引也有安IJ 索引，它可以被看做由Series组成的字典（共用同一个索引）。
#构建方法   最常用的一种是直接传入一个等长列表或NumPy数组组成的字典：

data= {'state':['Ohio','Ohio','Ohio','Nevada','Nevada'],
       'year':[2000,2001,2002,2001,2002],
       'pop':[1.5,1.7,3.6,2.4,2.9]}
DataFrame(data)   #结果会自动加上索引

frame2=DataFrame(data,columns=['year','state','pop','debit'],index=['one','two','three','four','five'])  #可以指定列顺序,   如果传入的列在数据中找不到， 就会产生NA值
frame2.columns

#通过类似字典标记的方式或属性的方式， 可以将DataFrame的要lj寂取为一个Series :
frame2['state']
frame2.state
frame2.ix['three'] #freame2.ix['three']
frame2.ix[0,:]   
frame2.ix[0][:] 

frame2['eastern'] = frame2.state == 'Ohio'     #为不存在的列赋值会创建出一个新列 

del frame2['debit']   #删除列
#通过索引返回的列只是相应数据的视图而已，并不是副本，因此，对返回的SeriesJ所做的任何就地修改全部会反映到源DataFrame上。通过Series的copy方法可显式地复制列
frame2.index.name='year'
frame2.columns.name='state'
frame2



# In[ ]:




# In[239]:

#索引对象   pandas 的索引对象负责管理轴标签和其他元数据（比如轴名称等）。构建Series或DataFrame肘，所用到的任何数组或其他序列的标签都会被转换成一个Index:
#Index对象是不可修改的（ immutab le），因此用户不能对其进行修改：
index = pd.Index(np.arange(3))
index
obj2 = Series([1.5, -2.5, 0], index=index)
obj2 

#Period Index 针对Period数据（时间间隔）的特殊Index


# In[217]:

#Index的方法和属性
方法说明
append 连接另一个Index对象产生一个新的Index
diff   计算差集,并得到一个Index
intersection 计算交集
union  计算并集
isin   计算一个指示各值是否都包含在参数集合中的布尔型数组
delete 删除索引i处的元素，并得到新的Index
drop   删除传入的值，并得到新的Index
insert 将元素插入到索引l处，并得到新的Index
is_monotonic 当各元素均大子等于前一个元素时，返回True
is_unique    当Index没有重复值时，返回True
unique       计算Index中唯一值的数组


# In[270]:

#重新索引   调用Series的reindex将会根据新索引进行重排。如果某个索引值当前不存在，就引入峡失值：
obj=Series([4.5, 7.2, -5.3, 3.6],index=['d','b','a','c'])
obj.reindex(['a','b','c','d','e','f'],fill_value=0 )#,method='ffill')
obj3=Series(['blue','purple','yellow'], index=[0,2,4])
obj3.reindex(arange(6),method='ffill')      #method='ffill'   实现前向值填充

#ffill或pad      前向填充（或搬运）值
#bfill或backfill 后向填充（或搬运）值


frame= DataFrame(np.arange(9).reshape((3, 3)), index=['a','c','d'],columns=['Ohio','Texas','California'])
frame
frame.drop('Texas',axis=1)    #删除列  
frame2 = frame.reindex([ 'a', 'b', 'c','d'])
frame2
#使用columns关键字即可重新索引列：
states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)
frame.reindex(index=['a', 'b','c', 'd'], method = 'ffill',columns=states)


# In[288]:

#索引、选取和过滤
obj4=Series(np.arange(4.), index=['a','b','c','d'])
obj4[obj<=2]
obj4[[ 'b', 'a', 'd']]
obj4[2:4]
obj4['b']
obj4[1]

obj4['b':'c']=5
obj4


# In[309]:

#对dataframe进行索引只能选取列，但有两种特例切片，布尔是选取行的
#对DataFrame进行索引其实就是藐取一个或多个列
data= DataFrame(np.arange(16).reshape((4, 4)),index=['Ohio', 'Colorado','Utah','New York'],columns=['one','two','three','four'])
data
data[[1,2]]
data['two']
#这种索引方式有几个特殊的情况。首先通过切片或布尔型数组选取行：
data[:2]
data[data['three']>5]

#另一种用法是通过布尔型DataFrame （比如下面这个由标量比较运算得出的）进行索引：
data<5
data[data < 5] = 0
data.get_value


# In[307]:

#为了在DataFrame的行上进行标签索引,我引入了专门的索引字段ix ,使你可以通过NumPy式的标记法以及轴标签从DataFrame中选取行和列的子集
data.ix['Colorado',['two','three']]   # 先行，后列
data.ix[2] 


#obj[val]                  选取DataFrame的单个列或一组列。在一些特殊情况下会比较便利：布尔型数组（过滤行）、切片（行切片）、布尔型Da taFrame （根据条件设置值
#obj.ix[val]               选取DataFrame 的单个行或一组行
#obj.ix[:,val]             选取单个列或列子集
#obj.ix[val1,val2]         同时选取行和列
#reindex方法               将一个或多个轴匹配到新索引
#xs方法                    根据标签选取单行或单列，并返回一个Series       
#icol 、irow方法           根据整数位置选取单列或单行，并返回一个Series 
#get_value、set_value方法  根据行标签和列标签选取单个值


# In[329]:

#算术运算和数据对齐
s1= Series([7.3,-2.5,3.4,1.5],index=[ 'a','c','d','e' ])
s2= Series([-2.1, 3.6, -1.5, 4, 3.1],index=['a','c','e','f','g'])            
s1+s2     #自动的数据对齐操作在不重叠的索引处引入了NA值 缺失值会在算术运算过程中传擂 ，对于DataFrame，对齐操作会同时发生在行和列上

df1 = DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),index=['Ohio', 'Texas', 'Colorado'])
df2 = DataFrame(np.arange(12.).reshape((4, 3)),columns=list('bde'),index=['Utah','Ohio', 'Texas','Oregon'])
df1+df2   #对齐操作会同时发生在行和列上

df1.add(df2,fill_value=0)

df1.reindex(columns=df2.columns,fill_value=0)

#add 用于加法（＋）的方法
#sub 用于减法（－）的方法
#div 用于除法（／）的方法
#mul 用于乘法（* ) 的方法
#Data Fram e和Series之间的运算     #frame.su b(series3, axis=O)
arr= np.arange(12.).reshape((3, 4))
arr - arr[0]
df1


# In[346]:

#函数应用和映射
np.abs(df1)
f = lambda x: x.max(axis=1) - x.min(axis=1)   #另一个常见的操作是， 将函数应用到由各列或行所形成的一维数组上。apply方法即可实现此功能：
f = lambda x: x.max() - x.min()
f(df1)

df1.apply(f)

def f(x) :
    return Series([x.min(), x.max()],index=[ 'min','max'])  #除标量值外，传递给apply的函数还可以返回由多个值组成的Seri es
df1.apply(f,axis=1)


format = lambda x: '%.2f' %x  # 元素级的Python函数也是可以用的。假如你想得到frame中各个浮点值的格式化字符串，使用applymap即可：
df1.applymap(format)           #之所以叫做applymap ，是因为Series有一个用于应用元亲级函数的map方

df1['b'].map(format)           # map应用到元素级



# In[375]:

#排序和排名  根据条件对数据集排序（ sorting ）也是一种重要的内置运算。要对行或列索引进行排序 （按字典l顺序），可使用sort index方毡，它将返回一个已排序的新对象

#对索引排序
obj = Series(range(4), index=['d','a','b','c'])
obj.sort_index()

frame= DataFrame(np.arange(8).reshape((2, 4)), index=['three','one'],columns=[ 'd', 'a','b','c'])
frame.sort_index()              #默认axis=0  行  。axis=1 列
frame.sort_index(axis=1,ascending=False)
#对值排序
obj = Series([4, 7, -3 , 2])
obj.sort_values()               #.order() 不使用了   在排序肘，任何触失值默认都会被放到Series的末尾：

#在DataFram 巳上，你可能希望根据一个或多个列中的值进行排序。将一个或多个列的名字传递给by选项即可达到该目的：
frame= DataFrame({'b':[4, 7, -3, 2],'a':[0,1,0,1]})
frame.sort_values(by=['a','b'])


#Series和DataFrame的rank方法
obj= Series([7, -5 , 7, 4, 2, 0, 4])
obj.rank(method='first',ascending=False,axis=0)                  #method='first' ,average  min,max,first

#带有重复值的轴索引
obj = Series(range(5),index=[ 'a','a','b','b','c'])

obj.index.is_unique   #False  索引的is_unique属性可以告诋你色的值是否是唯一的：
obj['a']              #对于带有重复值的索引，数据选取的行为将会有些不同。如果某个索引对应多个值，则返回一个Series g 而对应单个值的，则返回一个标量值
obj['c']


# In[385]:

#汇总和描述性统计

df=DataFrame([[1.4,np.nan],[7.1,-4.5],[np.nan,np.nan],[0.75,-1.3]],index=['a','b','c','d'],columns=['one','two'])
df.sum(axis=1,skipna=False)   #NA值会自动被排除，除非整个切片（这里指的是行或列）都是NA 。通过skipna选项可以禁用该功能
# 约简万洁的选顶
# axis 约简的轴。Data Frame 的行用0 ，列用1
# skipna 排除缺失值，默认值为True
# level 如果轴是层次化索引的（ 即Multilndex ），则根据level分组约简

#有些方怯(如idxmin和idxmax)返回的是间接统训（比如达到最小值或最大值的索引号）
df.idxmax()

#累计
df.cumsum()
#描述统计方法
df.describe()

#方法说明
#count     非NA值的数量
#describe  针对Series或各Data F ram e列计算汇总统计
#min 、max 计算最小值和最大值
#argmin 、argmax 计算能够获取到最小值和最大值的索引位置（ 整数）
#idxmin 、idxmax 计算能够获取到最小值和最大值的索引值
#quantile 计算样本的分位数（ 0到1 )
#sum      值的总和
#mean     值的平均数
#median   值的算术中位数（ 50%分位数）
#mad      根据平均值计算平均绝对离差
#var      样本值的方差
#std      样本值的标准差
#skew     样本值的偏度（三阶矩）
#kurt     样本值的峰度（四阶矩）
#cumsum   样本值的累计和
#cummin 、cummax样本值的累计最大值和累计最小值
#cumprod  样本值的累计积
#diff计   算一阶差分（对时间序列很有用）
#pct_change计算百分数变化


# In[ ]:

#相关系数与协方差
import pandas.io.data as web

all_data = {}
for ticker in ['AAPL','IBM','MSFT','GOOG']:
    all_data[ticker] = web.get_data_yahoo(ticker,'1/1/2000','1/1/2010')
    
#price=  DataFrame({tic: data['Adj Close'] for tic,data in all_data.iteritems()})                                 
#volume= DataFrame({tic:data['Volume'] for tic, data in all_data.iteritems()})


# In[457]:


import sys
import imp
imp.reload(sys)
#sys.setdefaultencoding('utf-8')
#sys.getdefaultencoding('ascii') 
#all_data=np.load('ticker.npy',encoding='utf-8')


# In[555]:

a=np.load('tricker.npz')
a.files
price=DataFrame(a['p1'])
volume=DataFrame(a['p2'])

returns = price.pct_change()
type(returns)
returns.columns=['AAPL','GOOG','IBM','MSFT']
volume.columns=['AAPL','GOOG','IBM','MSFT']
#returns

returns.dropna(axis=0,how='all')

returns.IBM.head()
returns.ix[:,2].corr(returns.ix[:,3])

returns.corr()

returns.cov()
returns.corrwith(returns.IBM)   #
returns.corrwith(volume)    # 按列名匹配


# In[241]:

#唯一值、值计数以及成员资格
obj = Series(['c','a','d ','a','a','b','b','c','c'])
obj.unique()
obj.value_counts(sort=False)
obj.isin(['b','c'])


# In[584]:

import pandas as pd

data=DataFrame({'qu1':[1,3,4,3,4],
                'qu2':[2,3,2,2,3],
                'qu3':[1,5,2,4,5]})
result = data.apply(pd.value_counts).fillna(0)
result


# In[594]:

#处理缺失数据
string_data = Series(['aardvark','artichoke',np.nan,'avocado'])

string_data.isnull()
string_data[0] = None
string_data
string_data.isnull()


#dropna   根据各标签的值中是否存在缺失数据对轴标签进行过滤，可通过阐值调节对缺失值的容忍度
#fillna   用指定值或插值方法（如ffi11或bfiLL）填充缺失数据
#isnull   返回一个含有布尔值的对象，这些布尔值表示哪些值是缺失值／NA ，该对象的类型与源类型一样
#notnull isnull 的否定式

from numpy import nan as NA
data = Series([1, NA, 3.5, NA, 7])
data.dropna(how='all')
data[data.notnull()]


# In[615]:

df = DataFrame(np.random.randn(7, 3))
df.ix[:4,1] = NA ; df.ix [:2,2] = NA
#df.fillna(0)
df.fillna({1:0.5,2:-1})   #通过一个字典调用flllna ， 就可以实现对不同的列填充不同的值：
#df.dropna(thresh=2)      #控制非缺失值的个数
#df


#flllna默认会返回新对象， 但也可以对现有对象进行就地修改：
#  _=df.fillna(0,inplace=True)

df.fillna(method ='bfill',limit=2)  #ffill
df.fillna(data.mean())

#value    用于填充缺失值的标量值或字典对象
#method   插值方式。如果函数调用时来指定其他参数的话，默认为 “ ffill”  bfill
#axis     待填充的轴，默认axis=O
#inplace  修改调用者对象而不产生副本
#limit    （对于前向和后向填充）可以连续填充的最大数量




# In[636]:

#层次化索引   层次化索引（ hierarchical index ing ）是pandas的一项重要功能，它使你能在一个轴上拥有多个（两个以上）索引级别。抽象点说，它使你能以低维度形式处理高维度数据。


data= Series(np.random.randn(10),index=[[ 'a','a','a','b','b','b','c','c','d','d'],[1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])
data.index

data['a']

data['a':'c']
data.ix[['b', 'd']]
#有时甚至还可以在“内层”中进行选取
data[:,2]
#层次化索引在数据重塑和基于分组的操作（如透视表生成）中扮演着重要的角色。比如说，这段数据’可以通过其uns tack方站被重新安排到一个DataFrame 中：
data.unstack()
data.unstack().stack()


#对于一个DataFrame ，每条轴都可以有分层索引：
data1= DataFrame( np.random.randn(9).reshape((3,3)),index=[[ 'a','b','b'],[1, 1, 2]],columns=[['a','b','c'],[1,2,3]])
data1.index.names = ['key1','key2']
data1.columns.names= ['state','color']
data1.index.name='key'
data1.columns.name='st'
data1


# In[642]:

#重排分级顺序
#有时，你需要重新调整某条轴上各级别的顺序，或根据指定级别上的值对数据进行排序。swaplevel接受两个级别编号或名称，并返回一个互换了级别的新对象（但数据不会发生变化）

data1.swaplevel('key1','key2')
#而sortlevel则根据单个级别中的值对数掘进行排序（稳定的） 。交换级别时，常常也会用到 sortlevel ，这样最终结果就是有序的了：

data1.sortlevel(0)

#注意： 在层次化索号｜的对象上，如果索引是按字典方式从外到内排序（目r1调用sortlevel（0） 或sort_index()的结果），数据选取操作的性能要好很多。

#根据级别汇总统计
data1.sum(level= 'key2')   #frame.sum(level='color’, axis=l)


# In[242]:

#使用DataFrame 的列    列转索引 或者索引转列
#人们经常想要将DataFrame的一个或多个列当做行索引来用， 或者可能希望将行索引变成DataFrame的列。以下面这个DataFrame为例：
import pandas as pd

from  pandas import DataFrame

fame=DataFrame({'a':range(7),'b':range(7,0,-1),'c':['one','one','one','two','two','two','two'],'d':[0,1,2,0,1,2,3]})

fame2=fame.set_index(['c','d'])#,drop=False)    #默认情况下， 那些列会从DataFrame中移除， 但也可以将其保留下来：drop=False
fame2.reset_index(['c','d'])   #索引变回列
#fame2


# In[678]:

#数据加载、存储与文件格式   P178


#read_csv       从文件、URL 、文件型对象中加载带分隔符的数据。默认分隔符为逗号
#read_table     从文件、UR L 、文件型对象中加载带分隔符的数据。默认分隔符为制表符（"\t"）
#read_fwf       读取定宽列格式数据（也就是说，没有分隔符）
#read_clipboard 读取剪贴板中的数据，可以看做read_table的剪贴板版。在将网页转换为表格时很有用


get_ipython().system('cat cho6/ex1.csv')
pd.read_table ('cho6/ex1.csv', sep=',')
pd.read_csv('ch06/ex2.csv', header=None)
pd.read_csv('cho6/ex2.csv', names=['a', 'b','c ','d', 'message'])
pd.read_csv('ch06/ex2.csv', names=names, index_col='message')
pd.read_csv('cho6/csv_mindex.csv', index_col=(' key1 ', 'key2'])
result= pd.read_table('ch06/ex3.txt',sep＝'\s+')
pd.read_csv('ch06/ex4.csv', skiprows=[O, 2, 3])          
result=pd.read_csv('cho6/exs.csv', na_values=('NULL'))
pd.read_csv('cho6/ex6.csv', nrows=5)              
#逐块读取文本文件    Text Parser还有一个get_chunk方桂，它使你可以读取任意大小的
chunker=pd.read_csv('cho6/ex6.csv',chunksize=lOOO)
tot= Series([])
for piece in chunker:
            tot = tot.add(piece['key'].value_counts(), fill_value=O)   
tot = tot.order(ascending=False)
#将数据写出到文本格式    利用Dataframe的to_csv方法，我们可以将数据写到一个以逗号分隔的文件中：
data.to_csv('cho6/out.csv')
data.to_csv(sys.stdout, sep='|')
data.to_csv(sys.stdout, na_rep='NULL')
data.to_csv(sys.stdout, index=False, header=False)    #指定列   cols=['a','b']
#Series也有一个to_csv方法：   
dates=pd.date_range ( '1/1/2000', periods=7)
ts=Series(np.arange(7),index=dates)
ts.to_csv('cho6/tseries.csv')
Series.from_csv('ch06/tseries.csv',parse_dates=True)
            


# In[ ]:

#JSON数据
import json
result = json.loads(obj)
相反，json.dumps 则将Python对象转换成JSON格式：

#读取Microsoft Excel 文件
xls_file = pd.ExcelFile ('data.xls')
table= xls_file.parse('Sheet1')   
#使用HTML和Web API    P192
import requests
url ='http://search . twitter.com/search.json?q=python%2opandas'
resp = requests.get(url)

import json
data = json.loads(resp.text)
data.keys()

tweet_fields = ['created_at','from_user ','id','text' ]
tweets= DataFrame(data['results'], columns=tweet_fields)


#使用数据库    
import sqlite3     #p195


# In[252]:

#数据规整化：清理、转换、合并、重塑
#合并数据集
#pandas.merge 可根据一个或多个键将不同DataFrame中的行连接起来
#pandas.concat可以沿着一条轴将多个对象堆叠到一起
#实例方法 combine_first可以将重复数据编接在一起，用一个对象中的值填充另一个对象中的缺失值
#数据库风格的DataFrame合并

df1 = DataFrame({'key': [ 'b', 'b', 'a', 'c', 'a','a','b'],'data1': range(7)})
df2 = DataFrame({'key':[ 'a', 'b', 'd'],'data2': range(3)})


pd.merge(df1,df2)    #注意，我并设有指明要用哪个列进行连接。如果没有指定， merge就会将重叠列的列名当做键。不过，最好显式指定一下：
pd.merge(df1,df2,on='key')    #如果列名不同   left_on =''  , right_on=''  指定不同的列， 默认 how =inner  left ,right,outer

#pd.merge(df1,df2 ,on= '' ,how='outer') 

#pd.merge(left, right,on ＝('key1','key2') , how='outer')  根据多个键合并

#对于合并运算需要考虑的最后一个问题是对重复列名的处理。虽然你可以手工处理列名重叠的问题（稍后将会介绍如何重命名轴标签），但merge有一个更实用的suffixes选项，
#用于指定附加到左右两个DataFrame对象的重叠列~上的字符串：

#pd.merge(left, right,on ＝'key1'，suffixes=('_left','_right')) 


#left_index    将左侧的行索引用作其连接键
#right_index   类似于left_index
#sort=True /False
#copy设置为False ，可以在某些特殊情况下避免将数据复制到结果数据结构中。默认总是复制


# In[305]:

#索引上的合并

left1 = DataFrame({'key':['a', 'b', 'a','a', 'b', 'c'],'value': range(6)})               
right1= DataFrame({'group_val':[3.5,7]}, index=['a','b']) 
left1
right1
pd.merge(left1,right1,left_on='key',right_index=True)


left1.join(right1, on='key')   #同时可以连接多张表[a,b],（早期版本的pandas) , DataFrame的join方法是在连接键上做左连接

right1.join(left1, how='outer')    #按索引连接  DataFrame还有一个join实例方怯，它能更为方便地实现按索引合并。它还可用于合并多个带有相同或相似索引的DataFram巳对象，
                                   #而不管它们之间有没有重叠的列。在上面那个例子中，我们可以编写

#有时候，DataFrame 中的连接键位于其索引中。在这种情况下，你可以传入left_index=True或right_index=True （或两个都传）以说明索引应该被用作连接键：
#merge(left, right, how='inner', on=None, left_on=None, right_on=None,      left_index=False, right_index=False, sort=True,      suffixes=('_x', '_y'), copy=True, indicator=False)


# In[327]:

#轴向连接
#另一种数据合并运算也被称作连接（concatenation) 绑定（binding）或堆叠(stacking ）。NumPy有一个用于合并原始NumPy数组的concatenation 函数：

arr= np.arange(12).reshape((3, 4))
np.concatenate([arr,arr], axis=0)

#pandas的coneat 函数提供了一种能够解决这些问题的可靠方式。我将给出一些例子来讲解其使用方式。假设有三个没有重叠索引的Series :

s1 = pd.Series([0,1], index=['a','b'])
s2 = pd.Series([2, 3, 4] ,index=[ 'c','d','e'])
s3 = pd.Series([5, 6], index=[ 'f','g' ])
s4 = pd.concat([sl * 5, s3])
pd.concat([s1, s2, s3])   #默认情况下， con cat是在axis=O上工作的，最终产生一个新的Series 。如果传入axis=l则结果就会变成一个DataFrame (axis=l是列）

pd.concat([s1, s2, s3],axis=1) 
pd.concat([s1, s4], axis=1)
pd.concat([sl, s4], axis=1,join='inner')


pd.concat([sl, s1, s3], keys=[ 'one', 'two','three' ],join='outer')



# In[335]:

#合并重叠数据

#还有一种数据组合问题不能用简单的合并（ merge ）或连接（concatenation ）运算来处理。比如说，你可能有索引全部或部分重叠的两个数据集。给这个例子增加一点启发性，
#我们使用NumPy的where函数，它用于表达一种矢；虽化的if-else :



a = pd.Series([np.nan, 2.5,np.nan,3.5,4.5,np.nan],index=['f','e','d','c','b','a'])
b = pd.Series(np.arange(len(a),dtype=np.float64),index=['f','e','d','c','b','a'])

b[-1] = np.nan

np.where(pd.isnull(a), b, a)

#Series有一个combine_first方法，实现的也是一样的功能，而且会进行数据对齐：

b[:-2].combine_first(a[2: ])







# In[377]:

#计算指标／哑变量    p226
df = DataFrame({'key': ['b','b','a','c ','a','b'],'data1': range(6)})


dummies= pd.get_dummies(df['key'], prefix='key')  

df[['data1']].join(dummies)
#df_with_dummy = df[['data1']].join(dummies)

#一个对统计应用有用的脑诀是： 结合get_dummies和i者如cut之类的离散化函数。


# In[466]:

#Pandas透视表（pivot_table）详解

import pandas as pd
import numpy as np

df = pd.read_excel("../in/sales-funnel.xlsx")
df.head()

df["Status"] = df["Status"].astype("category")
df["Status"].cat.set_categories(["won","pending","presented","declined"],inplace=True)


pd.pivot_table(df,index=["Name"])

pd.pivot_table(df,index=["Name","Rep","Manager"])

pd.pivot_table(df,index=["Manager","Rep"],values=["Price"],aggfunc=np.sum)

pd.pivot_table(df,index=["Manager","Rep"],values=["Price"],aggfunc=[np.mean,len])

pd.pivot_table(df,index=["Manager","Rep"],values=["Price"],columns=["Product"],aggfunc=[np.sum])
pd.pivot_table(df,index=["Manager","Rep"],values=["Price"],columns=["Product"],aggfunc=[np.sum],fill_value=0)
pd.pivot_table(df,index=["Manager","Rep"],values=["Price","Quantity"],columns=["Product"],aggfunc=[np.sum],fill_value=0)

pd.pivot_table(df,index=["Manager","Rep","Product"],values=["Price","Quantity"],aggfunc=[np.sum],fill_value=0)
pd.pivot_table(df,index=["Manager","Rep","Product"],
               values=["Price","Quantity"],
               aggfunc=[np.sum,np.mean],fill_value=0,margins=True)


pd.pivot_table(df,index=["Manager","Status"],values=["Price"],
               aggfunc=[np.sum],fill_value=0,margins=True)

pd.pivot_table(df,index=["Manager","Status"],columns=["Product"],values=["Quantity","Price"],
               aggfunc={"Quantity":len,"Price":np.sum},fill_value=0)


table = pd.pivot_table(df,index=["Manager","Status"],columns=["Product"],values=["Quantity","Price"],
               aggfunc={"Quantity":len,"Price":[np.sum,np.mean]},fill_value=0)
table

#高级透视表过滤

table.query('Manager == ["Debra Henley"]')
table.query('Status == ["pending","won"]')
#http://python.jobbole.com/81212/


# In[468]:

frame=pd.DataFrame(np.random.randn(3,4))

frame[1]=frame[1].astype("category")
frame[1].cat.set_categories( ["-0.445380" , "-0.580579"  ,"-1.507094"],inplace=True) 


# In[ ]:

0   -0.580579
1   -1.507094
2   -0.445380


# In[348]:

frame


# In[474]:

#绘图和可视化
get_ipython().magic('matplotlib inline')
import  matplotlib.pylab as plt    #import  matplotlib.pyplot as plt
#import matplotlib as plot
#plt.plot(np.arange(10))

# matplotlib的图像都位于Figure对象中。你可以用plt.figure创建一个新的Figure：    

fig=plt.figure()             #创建一个Figure对象 
fig.add_subplot(2,2,1)       #由fig.add_subplot 所返回的对象是AxesSubplot对象， 直接调用它们的实例方能在其空格里画图了
fig.add_subplot(2,2,2) 
fig.add_subplot(2,2,3)
fig.add_subplot(2,2,4)
#plt.figure(2).add_subplot(1,1,1)    #创建另一个figture对象
#plt.gcf()     #plt.gcf即可得到当前Figure的引用。

plt.plot([1.5,3.5,-2,1.6])       


from numpy.random import randn
import numpy as np
fig.add_subplot(2,2,2).plot(randn(50).cumsum(),'k--')    #"k--"是一个线型选项，用于告诉malplotlib绘制黑色虚线图

_ = fig.add_subplot(2,2,1).hist(randn(100), bins=20, color='b',alpha=0.8)

fig.add_subplot(2,2,3).scatter(np.arange(30),np.arange(30) + 3 * randn(30))


# In[51]:

#你可以在matplo tlib的文柏中找到各种图表类型。由于根据特定布局创建Fi gure和subplot是一件非常常见的任务，于是便出现了一个更为方便的方洁（ plt.subplots ），
#它可以创建一个新的Figure ,并返回一个含有已创建的subplot对象的Nun1Py数组：
fig,axes=plt.subplots(2,3,sharey='all', sharex='all')
axes[0,1]

#pyplot.subplots的选项
#nrows       subplot的行数
#ncols       subplot的列数
#sharex      所有subplot应该使用相同的×轴刻度（调节xlim将会影响所有subplot )
#sharey      所有subp lot应该使用相同的Y轴刻度（调节ylim将会影响所有subplot )
#subplot_kw  用于创建各subplot的关键字字典
#**fig_kw    创建figure时的其他关键字，如plt.subplots ( 2, 2, figsize＝(8,6))

axes[0,1].plot(np.arange(10).cumsum())


# In[475]:

#调整subplot周围的间距,利用Figure的subplots_adjust方法可以轻而易举地修改间距，此外，它也是个顶级函数
#subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None,hspace=None)


fig,axes= plt.subplots(2, 2, sharex=True, sharey=True)
for i in range(2):
    for j in range(2):
        axes[i, j].hist(randn(500), bins=50, color='g', alpha=1)
plt.subplots_adjust(wspace=0, hspace=0)


# In[65]:

x,y,z=plt.hist(randn(500), bins=50, color='g', alpha=1)


# In[98]:

#颜色、标记和线型
x=np.arange(30) 
y=np.random.randn(30)
plt.plot(x, y, 'g*--',drawstyle='Default', label='Default')       #plt.plot(x, y,linestyle＝'一',color='g')   
                             # 线型图还可以加上一些标记（ marker）   plt.plot(randn(30).cumsum(), ’ ko--')  ，但标记类型和线型必须放在颜色后面
plt.plot(randn(30).cumsum(),color='r',linestyle='dashed', marker='o',drawstyle='steps-post' ,label='step-post')
plt.legend(loc='best')           #添加图例

#如xlim，xticks和xticklabels之类的方怯,在们分别控制图表的范围、刻度位置、刻度标签等
plt.xlim()         #返回当前的X轴绘图范围。 
plt.ylim()         
plt.xlim([0,30])   #将X轴的范围设置为0到l0


# In[108]:

#设置标题、轴标签、衷lj度以及刻度标签

fig = plt.figure();ax = fig.add_subplot( 1,1,1)

ax.plot(randn(1000).cumsum())

#要修改X轴的刻度，最简单的办律是使用set_xticks和set_xticklabels
ticks = ax.set_xticks([0, 250, 500, 750, 1000])
labels =ax.set_xticklabels(['one','two','three', 'four','five'],rotation=30, fontsize='small')

ax.set_title('My first matplotlib plot')    #添加标题
ax.set_xlabel('Stages')           #添加轴标签


# In[132]:

#添加图例
fig = plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(randn(1000).cumsum(),'g', label='one')
ax.plot(randn(1000).cumsum(),'r--', label='two')
ax.plot(randn(1000).cumsum(),'b.', label='three')
ax.legend(loc='best')   #loc告诉m atplotli b要将困例放在哪。如果你不是吹毛求疵的话，“ beat”是不错的选择，因为它会选择最不碍事的位置。
                        #要从图例中去除一个或多个元素，不传人label或传入label='_nolegend_'  
#loc
#center left
#center
#lower left
#upper left
#lower right
#upper right
#center right
#best
#upper center
#lower center
#right


#加标注
ax.annotate('xxxxxxxx'               #  要显示的内容
           ,xy=(300,20)              #  显示的坐标
           ,xytext=(300,50)          #  显示的文本坐标
           ,arrowprops=dict(facecolor='g')   #箭头颜色等配置
           ,horizontalalignment='left'       #水平对齐方式
           ,verticalalignment='top'          #垂直对齐方式
           )
           
ax.annotate('yyyyyy'               #  要显示的内容
           ,xy=(500,10)              #  显示的坐标
           ,xytext=(150,30)          #  显示的文本坐标
           ,arrowprops=dict(facecolor='g')   #箭头颜色等配置
           ,horizontalalignment='left'       #水平对齐方式
           ,verticalalignment='top'          #垂直对齐方式
           )


#for date, label in crisis_data:
#    ax.annotate(label,xy=(date, spx.asof(date) + 50),xytext=(date,spx.asof(date) + 200),arrowprops=dict(facecolor='black'),horizontalalignment='left',verticalalignment='top')


    
    
    


# In[123]:

#注释 annotate

from datetime import datetime
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
data=pd.read_csv('ch08/spx.csv ',indexcol=0, parse_dates=True)
spx = data['SPX']
spx.plot(ax=ax,style='k-')
crisis_data=[ (datetime(2007,10,11),'peak of bull market'),
              (datetime(2008,3,12),'bear stearns fails'),
              (datetime(2008,9,15),'lehman bankruptcy')
         
for date, label in crisis_data:
    ax.annotate(label,xy=(date, spx.asof(date) + 50),xytext=(date,spx.asof(date) + 200),arrowprops=dict(facecolor='black'),horizontalalignment='left',verticalalignment='top')

ax.set_xlim(['1/1/2007' , '1/1/2011'])
ax.set_ylim( [600, 1800])
ax.set_title('Important dates in 2008-2009 financial crisis') 


# In[143]:

#图形的绘制要麻烦一些。matplotlib有一些表示常见图形的对象。这些对象被称为块(patch）。
#其中有些可以在matplotlib.pyplot 中找到（如Rectangle和Circ时，但完整集合位于matplotlib.patches

fig= plt.figure()
ax =fig.add_subplot(1, 1, 1)

rect =plt.Rectangle((0.2,0.75),0.4,0.15,color='k',alpha=0.1)
circ =plt.Circle((0.7,0.2),0.15,color='b',alpha=0.3)
pgon = plt.Polygon([[0.15,0.15],[0.35,0.4],[0.2,0.6]],color='g',alpha=0.5)
ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)


# In[485]:

plt.Rectangle((0.2,0.75),0.4,0.15,color='k',alpha=0.1)
show()






# In[145]:

#将图表保存到文件.
#plt.savefig ('figpath.svg')
plt.savefig('figpath.png', dpi=400, bbox_inches='tight')    #是dpi（控制“每英寸点数”分辨率）和bbox_inches(可以剪除当前图表周围的空白部分）。
                                                            #要得到一张带有最小白边且分辨率为400DPI的PNG 图片


# In[146]:

#matplotlib配置   p255


# In[152]:

#pandas 中的绘图函数 
#matplotlib实际上是一种比较低级的工具。要组装一张图标，你得用它的各种基础组件才行
#pandas有许多能够利用DataFrame对象数据组织特点来创建标准图表的高级绘图方住（这些函数的数量还在不断增加）


#线型图  Series和DataFrame都有一个用于生成各类图表的plot方怯。默认情况下，它们所生成的是钱型图
import pandas as pd

s = pd.Series(np.random.randn(10).cumsum(), index=np.arange(0,100,10))
s.plot()

#该Series对象的索引会放传给matplotlib,用以绘制X铀。可以通过use_index = False禁用该功能，X轴刻度和界限可以通过xticks和xlim选项进行调节，Y轴就用yticks和ylim

#Series.plot方法的参数
#label  用于图例的标签
#ax     要在其上进行绘制的matplotlib subplot对象。如果没有设置，则使用当前matplotlib subplot
#style  将要传给matplotl ib的风格字符串（如’ko--')
#alpha  图表的填充不透明度（0至1之间）
#kind   可以是'line','bar', 'barh','kde’
#logy   在Y轴上使用对数标尺
#use_index  将对象的索引用作刻度标签
#rot    旋转刻度标签（ 0到360 )
#xticks 用作X轴刻度的值
#yticks 用作Y轴刻度的值
#xlim   X轴的界限（例如［0, 10] )
#ylim   Y轴的界限
#grid   显示轴网格线（默认打开）


#专用于DataFrame的plot的参数
#subplots 将各个Data Fram e列绘制到单弛的subplot 中
#sharex   如果s ubplots=True ，则共用同一个X轴，包括刻度和界限
#sharey   如果subplots=True ，则共用同一个Y轴
#figsize      表示图像大小的元组
#title        表示图像标题的字符串
#legend       添加－个subplot 图例（默认为True)
#sort_columns 以字母表顺序绘制各列，默认使用当前列顺序

#stacked=True  堆叠


# In[490]:

#柱状图

fig , axes = plt.subplots(2, 1)
data= pd.Series(np.random.rand(16), index=list('abcdefghijklmnop'))
data.plot( kind='bar', ax=axes[0], color='k', alpha=0.7)
data.plot(kind='line', ax=axes[0], color='r',alpha=0.7)




# In[164]:

data.plot(kind='barh', stacked=True, alpha=0.5)
#data.value_counts().plot(kind='barh')


# In[ ]:

#直方图和密度图   tips( 'tip_pct') .hist(bins=50)    tips['tip_pct'].plot(kind='kde')   密度图也被称作KDE (Kerne l Density Estimate,核密度估计）图。调用plot时加上kind ＝’ kde ’ 即可生成一张密度图（标准棍合正态分布



#散点图   plt.scatter( trans_data [’mt'), trans_data [ ’ unemp’ ])
#Data Fram e创建散布图矩阵的scatter matrix 函数。它还支持在对角线上放置各变量的直方图或密度图
#pd.scatter_matrix(trans_data, diagonal='kde', color='k', alpha=0 .3)


# In[ ]:




# In[184]:

#数据聚合与分组运算
#根据一个或多个键（可以是函数、数组或DataFrame列名）拆分pandas对象
#计算分组摘要统计，如计数、平均值、标准差，或用户自定义函数
#对DataFrame的列应用各种各样的函数
#应用组内转换或其他运算， 如规格化、线性回归、排名或选取子集等
#计算透视表或交叉衰
#执行分位数分析以及其他分组分析

#groupby技术       split-apply-combine  （拆分一应用一合并）


df = pd.DataFrame({'key1': ['a','a ','b', 'b','a'] ,
                   'key2': ['one ','two' , 'one' ,'two','one'] ,
                   'data1':np.random.randn(5),
                   'data2':np.random.randn(5)})



grouped=df['data1'].groupby(df['key1'])     #GroupBy对象但没有进行计算，只是记录了分组运算所需要的一切信息
grouped.mean()


means= df['data1'].groupby([df['key1'], df['key2']]).mean()

means.unstack()


states = np.array([ 'Ohio', 'California','California', 'Ohio','Ohio'])
years= np.array([2005, 2005, 2006, 2005, 2006])

df['data1'].groupby([states, years]).mean().unstack()    #  非数值列会自动排除  KEY2

df.groupby(['key1','key2']).size().unstack()    #计算分组大小





# In[215]:

#对分组进行迭代    #GroupBy对象支持选代,可以产生一组二元元组（由分组名和数据块组成）
for name, group in df.groupby('key1'):
    print(name)
    print(group)
pieces= dict(list(df.groupby('key1')))
pieces['b']


#选取一个或一组列
df.groupby('key1')['data1']
df.groupby('key1')[['data2']]    



# In[205]:

df.groupby('key1')['data1'].mean()      #这种索引操作所返回的对象是一个已分组的已分组的Series （如果传入的是标量形式的单个列名）
df.groupby('key1')[['data1']].mean()    #这种索引操作所返回的对象是一个已分组的DataFram e （如果传人的是列表或数组）


# In[219]:

#通过字典或Series进行分组
people=pd.DataFrame(np.random.randn(5,5),columns=['a','b','c','d','e'],index=['joe','steve','wes','jim','travis'])
people.ix[2:3, ['b','c']]=np.nan

people


# In[211]:

type(df['data1'])
#type(df[['data1']])


# In[ ]:




# In[ ]:



