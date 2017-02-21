
# coding: utf-8

# # step 1

# 

# In[23]:

#数字及表达式
from __future__ import division    #通过future可以导入那些在未来会成为标准Python组成部分的新特性
1/2     
1//2    #整除
1%2     #取余
2**3    #幂，乘方
-2**3==(-2)**3  # 一元运算符优先
pow(2,3)
abs(-3.5)
float()
help()
str()  #转换成字符串
repr() #返回值的字符串表示形式
round(1.0/2.0)
1000000000000000000000000000 * 5000000000000000000000000000000   #长整形  , 100000000000000000000L  PYTHON3 已经没有此种格式

# 2to3.py  python 2 转成3 的工具


# %cp ../../xlrao/bigtable.txt /xlzhong/Data_xl   复制数据命令



# In[45]:

#变量  变量名可以包括字母，数字，下划线。 变量不能以数字开头
#用户输入
input("the meaning of life:")   
raw_input()   # 会把说输入当做原始数据处理    py 3.0已经没有这种写法了， 统一用input

#长字符串   ''' like this  '''.
#原始字符串  r'c:\nowhere’
#unicode字符串 u'hello world'
#转义 \
'let\'s go' 
"let's go"
#字符串拼接
"hello." + "world"


# In[261]:

#模块
import math
from math import sqrt 
x=math.floor(32.9)
int(math.floor(3.2))    # floor ，ceil 向上取整
type(x)   #类型对象






# In[274]:

a=5 
print("hdhdhdh is " + repr (a) )
str(0.1)


greeting='hello'
greeting.count('h')


print("hdhdhdh is %s " %a )


# In[254]:

#复数
import cmath
cmath.sqrt(-1)


# In[246]:

#保存并运行Python
#以  .py 结尾
#运行  python  hello.py
#脚本首行 加上   #!/usr/bin/env python
#在实际运行脚本之前，必须让脚本文件具有可执行属性   $chmod a+x hello.py


# In[ ]:

print str("hello world!")


# In[3]:

#系列 
#数据结构  ；列表 元组
#通用系列操作  索引 分片 加 乘 及检查某个成员是否属于系列的成员 ，系列长度，最大元素，最小元素
greeting='hello'
greeting[0]    #索引 0 开始 ，倒过来则是-1 开始    greeting[-1] 则是取最后一位



# In[4]:

#索引
#example
months = ['january', 'february', 'march',  'april', 'may', 'june','july', 'august', 'september', 'october','november', 'december']
endings=['st','nd','rd'] + 17*['th'] +['st','nd','rd'] +7*['th']           +['st']
year  = input('year: ')
month = input('month (1-12): ')
day   = input('day(1-31): ')
month_number = int(month)
day_number   = int(day)
month_name = months[month_number - 1]
ordinal    = endings[day_number - 1]
print (month_name + ' '+ ordinal + '.' + year)


# In[6]:

#分片
months[2:5]
months[2:]
months[-3:]
months[2:11:2]    #开始：结束：步长
months[::2]
#系列相加 ,乘法 
[1,2,3]+[1,2,3]  #连接
'python'*5


# In[7]:

#example
sentence=input("sentence: ")
screen_width=80
text_width=len(sentence)
box_width=text_width + 6
left_margin =(screen_width -box_width )//2

print( )
print( ' '* left_margin + '+' + '-'*(box_width-6) + '+')
print( ' '* left_margin + '|' + ' '*(text_width) + '|')
print( ' '* left_margin + '|' +     (sentence) + '|')
print( ' '* left_margin + '|' + ' '*(text_width) + '|')
print( ' '* left_margin + '+' + '-'*(box_width-6) + '+')
print( )


# In[8]:

#成员资格
permissions = 'rw'
#type(permissions)
'w' in permissions
user =['mlh','foo','bar']
input("enter your user name: ") in user


# In[98]:

numbers = [100,34,678]
type(numbers)
min(numbers)
max()
len()


# In[15]:


x=[1,2,3]
x[0:]=list('15') 
x


# In[17]:

#列表  []   list()
list_a=list('help')  #list  ['h','e','l','p']
list_b='help'        #str
type(list_a)         #查看类型
type(list_b)         #查看类型
''.join(list_a)      #列表转成字符串 合并
list(list_b)         #字符串转成列表
#基本列表操作
x=[1,2,3]
x[1]=5
del x[2]             #删除元素
x[0:]=list('15')     #分片赋值
name=list('perl')
name[1:1]=list('ython')    #分片插入
name[1:4]=[]               #分片删除
del name[2:4]
#列表方法  方法是与某些对象紧密联系的函数 ，对象可能是列表，数字，字符串或其他对象    调用方式    对象.方法
lst=[1,3,4,5,7]
lst.append(5)            #直接修改原来的列表，不是生成新的列表
lst.count(1)             #统计某个元素在列表中出现的次数
lst.extend(lst)          #追加新的列表  ，跟连接 +  操作相似，但不会生成新的列表
lst.index(4)             # 查找第一个匹配的索引位置
lst.insert(3,'four')     #在索引位置插入新的值
lst.pop()                #移除列表最后一个元素  ， lst.pop(0) 第一个元素  ，唯一一个既修改列表 也返回元素值得列表方法   栈   后进先出
lst.append(lst.pop())    #后进先出
lst.insert(0,lst.pop(0)) #先进先出
lst.remove(5)            #移除首次匹配的值
lst.reverse()            #元素反向存放    #对系列反向迭代 reversed(lst)   生成一个反向迭代器，  list(reversed(lst))       
lst.sort()               # 修改原列表但返回为空    x=y, x=y[:]  两者的区别是x=y 指向同一列表，x=y[:] 指向不同列表
sorted(lst)              #生成新的列表   sorted(‘python’)  返回的也是列表   
#cmp(42,32)               # -1,0,1   compare(x,y)
x=['asfsdf','dsfd','fd','dsfasdee']
x.sort(key=len)          #x.sort(reverse=Ture）


# In[150]:

#元组：不可变系列   ，元组不能修改  需要添加逗号 ， tuple()
1,2,3
tuple([1,2,3])      #tuple 功能与list基本是一样的，以一个系列作为参数并把它转成元组   (1, 2, 3)    元组 （）
#元组操作
x=1,2,3,1,1,1,0
x[0:2]              #元组分片还是元组
x.count(1)
x.index(0)          #值所在的位置，索引


# In[167]:

#字符串    字符串不可变，不能分片赋值       import string as str
website='http://www.python.org'
website[0:4]
#字符串格式化 %
formats = "hello, %s. %s enough for ya?"     # %s 格式化字符串   %s转换说明符,如果其他部分包含% 则使用%%
values =('world','hot')
print(formats % values)

format = "Pi with three decimals: %.3f"      #带精度的  .3f
from math import pi
print( format % pi)
#模板字符串
from string import Template
s=Template('$x, glorious $x!')          #替换全部$x      ，如果其他部分包含$ ,则使用$$
s.substitute(x='slurm')

k=Template("it's ${x}tastic!")          #替换某一部分  ${x}
k.substitute(x='slurm')


# In[174]:

#字符串格式化   
#    %字符  标记转换说明符开始
#    转换标志    -表示左对齐，+表示转换值之前要加上正负号，“”表示正数之前保留空格，0表示转换位数不够则用0填充 ,- 左对齐
#    最小字段宽度
#    点（.）后跟精度       .3f,  .*s  .5s , *从后面的元组中读取  
#    转换类型   d 十进制正数 ， f 十进制浮点数， s 字符串   C  单字符  参考书 python基础教程 第二版 ，p64

'%s plus %s quuals %s' %(1,1,2)    # %() 元组要用括号
'price of eggs: $%d' %42
'%10.2f' % pi   #字段宽度 10 ， 精度 2 
'%.*s' %(5,'guido van rossum')


# In[186]:

#example
width =int(input('please enter width: '))        #  input得到的值类型是 str  若要当做数字用需要类型转换
price_width = 10
item_width = width - 10
header_format = '%-*s%*s'                    
format        = '%-*s%*.2f'                       #   -表示左对齐， * 位数 ， .2f 精度
print('=' * width)
print(header_format % (item_width,'item',price_width,'price'))
print('-' * width)
print(format %(item_width,'apples',price_width,0.4))
print('-' * width)


# In[215]:

#字符串方法   string.digits  .lettters  .lowercase  .printable  .punctuation  .uppercase
title="monty python's flying circus"    
title.find('monty')        # find 查找子字符串， 返回最最左边的位置
title.find('python',2,15)  #可以提供起始点，结束点   , 返回-1 表示未找到
seq = ['1','2','3','4','5'] 
sep = '+'
sep.join(seq)              # 只能连接字符型的     是split 的逆操作
'+'.join(seq)
'1+2+3+4+5'.split('+')     # 展开成列表
name='Rao'
name.lower()
'   rao   '.strip()        #去除两端的空格
'Rao xin'.replace('xin','le')
               
#from string import maketrans      #可能是老版本的， 再此行不通
#table = maketrans('CS','KZ')
#'THIS IS AN INCREDIBLE TEST'.translate(table)    #替换单个字符   translate   , 但可以多个同时替换    


# In[18]:

#字典 { }   dict()
name=['alice','beth','cecil','dee-dee','earl']
numbers=['2341','9102','3158','0142','5551']
numbers[name.index('cecil')]
#创建字典
phonebook = {'alice':'2341','beth':'9102','cecil':'3258'}
d=dict([('name','gumby'),('age',42)])
d['name']
dict(name='gumby',age=42)


# In[259]:

#字典操作
len(phonebook)
phonebook['alice']
phonebook['alice']=2345
#del phonebook['cecil']
'alice' in phonebook
x={}
x[42]='foobar'


# In[21]:

#字典方法
from copy import deepcopy
d={}
d['name']='gumby'
d['age']=42
d1=d.copy()              # 副本会改变
d3=deepcopy(d)           # 副本不会被改变   from copy import deepcopy
d2=d                     # 
#d.clear()
d3.get('name')           #  d3.get('acb','N/A')   get不存在的键是不会报错， 但 d['abc'] 则会报错
#fromkeys方法使用给定的键建立新的字典，每个键默认的值为none
{}.fromkeys(['name','age'],'(unknown)') # {}.fromkeys(['name','age'] )
'name' in d3
d3.items()               #返回所有的字典项
#d3.iteritems( )         #返回迭代器
d3.keys()                #获取键
d3.values()              #获取值
#d3.pop('name')          # 获取对应给定键的值，然后将这个键-值对从字典中移除
#d3.popitem()            #弹出随机项
d3.setdefault('name')    #类似get  
#d3.update(x)             #用一个字段更新另一个字典


# In[32]:

d3.get('name1','N/A') 
d3.setdefault('name1')


# In[295]:

#条件、循环和其他语句
# import somemodule   #从模块导入函数    from somemodule import somefunction,anotherfunction,yetantherfunction \ import somefunction  from  somemodule
#import math as foobar
# from math import sqrt as foobar


# In[35]:

#序列解包(sequnece unpacking)
x,y,z=1,2,3
print(x,y,z)
scoundrel ={'name':'robin','girlfriend':'narion'}
key,value = scoundrel.popitem()
print(key,value)

# a,b,rest* = [1,2,3,4] 

#莲式赋值（chained assignment）    ,x =y =sumefuncton()     # 不同于 x=sumefuncton() ,y =sumefuncton() 
#增强赋值   
x =2 
x = x + 1
x += 1
x *= 2
print(x)
#语句块
#条件和条件语句   下面的值在最为布尔表达式的时候会被解释为 假（false） False,None,0 "",(),{},[]  ,其他一切则被解释成真。
bool('i think, therefore i am')


# In[55]:

#条件执行 和if语句
name=input('what is your name ?')
if name.endswith('Gumby') :       # 方法：endswith  判断字符串的结束字符串是什么，startwith 开始支付串
   print('hello, Mr. Gumby')
#else 子句
else :
   print('hello, stranger')
#elif子句   （else if 的简写）
num = int(input('Enter a number: '))   #input 获取的是字符串，需要穿换成数值型
if num > 0:
    print('the number is positive')
elif num < 0 :                         #elif  不能使用 else if 代替
    print('the number is negative')
else :
    print('the number is zero')
#嵌套代码块   if 子句中嵌套 if   


# In[68]:

#更复杂的条件
#比较运算符
#x==y
#x!=y
#x is y   xy是同一对象，x is not y 
#x in y
#0 < age < 100
#实际顺序 与 locale 相关 ？

#if((cash >prince) or customer_has_good_credit) and  not out_of_stack:    # 复杂条件  or  not   and  
#    give_goods()

#短路逻和条件表达式  （短路逻辑 short-circuit logic，惰性求值 lazy evaluation）
name = input('please enter your name: ') or '<nuknown>'   
print(bool(name))
name

# a if b else c    #  根据B的真假 输出a 或者 c 相当于其他语言的三元运算符    if（bool ，a，b）
#print (  4  if  3 < 4 else  1 )
# 断言
#age = 10
#assert  0 < age < 100. 'the age must be realistic'   #?


# In[69]:

#循环
#while
x = 1 
while x <= 5:
    print(x)
    x+=1

#for 
words = ['this','is','an','ex','parrot']    
for word in words:
    print(word)

#迭代
#range(0,10) ,range(10) ,range(0,10,2)  #第三个参数表示步长
for number in range(0,10):              #能用for 就尽量少用while
    print(number)




# In[16]:

#循环遍历字典元素
d = {'x':1,'y':2,'z':3}
for key in d :
    print(key,'corresponds to ',d[key])
    
for key,value in d.items():
    print(key,'corresponds to ',value)
#一些迭代工具
#1 并行迭代
names=['anne','beth','george','damon']
ages=[12,45,32,102]

for i in range(len(names)):
    print(names[i],'is',ages[i],'years old')

zip(names,age)
for name,age in zip(names,ages):
    print(name,'is',age,'years old')
    


# In[22]:

#翻转和排序迭代   reversed sorted  ， 不是原地修改对象，而是返回翻转或排序后的版本，与sort 及reverse 不同,他们是修改原来的序列
sorted([4,3,6,8,3])
sorted('hello,world!')
list(reversed('hello,world!'))       # reversed 只返回一个可迭代的对象，如要对其进行操作则需要先将转换成list类型
''.join(reversed('hello,world!'))


# In[73]:

''.join((reversed('hello,world!')))


# In[74]:

#跳出循环 break
from math import sqrt
for n in range(99,0,-1):       #range(0,10,2)    0 2 4 6 8
    root = sqrt(n)
    if root ==int(root):
        print(n)
        break   #找到第一个就终止循环，也可以去掉，则会找出所有符合条件的n    


# In[ ]:

#continue
for x in seq:
    if condition1: continue
    if condition2: continue
    if condition2: continue  
    
    do_something()
    do_something_else()
    do_anohter_thing()
    etc()


# In[28]:

#while True/ break
while True:
    word=input('please enter a word: ')
    if not word:break
    print('the word was '+word)


# In[35]:

#列表推导式——轻量级循环
[x*x for x in range(0,10)]
[x*x for x in range(0,10) if x % 3 == 0]   #[0, 9, 36, 81]
[(x,y) for x in range(3) for y in range(3)]

result=[]
for x in range(3):
    for y in range(3):
        result.append((x,y))
result


# In[97]:

girls=['alice','bernice','clarice']
boys = ['chris','arnold','bob']
letterGirls={}
for girl in girls:
    letterGirls.setdefault(girl[0],[]).append(girl)  #setdefault([],[])  类似get, 但不同的是如果key不存在时会自动创建该key ,value 置成默认值，然后通过append添加values   
letterGirls   
print([b+'+'+g for b in boys for g in letterGirls[b[0]]] )


# In[89]:

letterGirls={}
letterGirls.setdefault(girl[0],[]).append(girls[0])
letterGirls


# In[103]:

#pass 
#del    x=None  ,del x   删除对象
#exec  eval  执行和求值字符串
exec("print('hello,world!')")
from math import sqrt
scope={}
exec("sqrt = 1",scope)     # scope 放置代码的命名空间   2.0语法，exec "sqrt = 1" in scope,  3.0语法  exec("sqrt = 1",scope) 
scope['sqrt']
sqrt(9)

#eval     exec语句会执行一系列python语句，不反回任何对象，而eval 会计算python表达式 （以字符串形式书写 ）并返回结果
eval(input("enter an arithmetic expression: "))     #2+5+6


# In[ ]:




# In[104]:

#抽象
fibs=[0,1]
for i in range(8):
    fibs.append(fibs[-2] + fibs[-1])
fibs  #斐波那契数列   [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]




# In[123]:

#函数
def  fibs(num):
    result=[0,1]
    for i in range(num-2):
        result.append(result[-2]+result[-1])
    return result

#del fibs()
fibs(7)       #del fibs()


# In[2]:

def square(x):
    'Calculates the square of the number x.'    #函数说明
    return x*x

square.__doc__   # 查看函数文档
help(square)     # help 可以得到关于函数，包括他的文档字符信息


# In[17]:

def init(data):                   #{'first': {}, 'last': {}, 'middle': {}}
    data['first']={}
    data['middle']={}
    data['last']={}

def lookup(data,label,name):
    return data[label].get(name)

def store(data,full_name):
    names=full_name.split()
    if len(names)==2: names.insert(1,'')          # 列表的该位置被新的值插入后,原来的值向后移，没有被覆盖掉
    labels='first','middle','last'
    for label,name in zip(labels,names):
        people=lookup(data,label,name)
        if people:
            people.append(full_name)
        else:
            data[label][name]=[full_name]
            
MyNames={}
init(MyNames)
store(MyNames,'Magnus Lie Hetland')
lookup(MyNames,'middle','Lie')
        


# In[68]:

def store(data,*full_names):                       #一次存储多个名字
    for full_name in full_names:
        names=full_name.split()
        if len(names)==2: names.insert(1,'')          #列表的该位置被新的值插入后没原来的值向后移，没有被覆盖掉
        labels='first','middle','last'
        for label,name in zip(labels,names):
            people=lookup(data,label,name)
            if people:
                people.append(full_name)
            else:
                data[label][name]=[full_name] 


# In[124]:

#函数参数  
#位置参数   store(a,b,10)
#关键字参数 store(a=10,b=20,c=10)  # **代表关键字参数
#收集参数   store(*params)    #  *表示收集其余位置参数
def print_params_4( x, y, z=3, *pospar, **keypar):
    print(x,y,z)
    print(pospar)
    print(keypar)

    
print_params_4(1,2,3,'d','c',a=3,b=6)


# In[116]:

#反转过程
def add(x,y): return x + y
params=(1,2)  
add(*params)             # *  元组  ，**字典


# In[105]:

#作用域
x = 1
scope = vars()      # vars() 可以返回作用域内的所有变量 ,是字典类型  命名空间，有全局作用域，每个函数调用时会创建新的作用域
scope['x']

#globals()['param']    #获取全局变量 ，字典类型 , 与 vars()类似
#locals()              #返回局部变量的字典


# In[162]:

key,value=vars().popitem()
print(key)
print(value)


# In[80]:

#递归


# In[ ]:

# map,filter,reduce     python在应对这类“函数式编程”方面有一些有用的函数(python3.0中这些度被移至functions模块中)

map(func,seq[,seq,.....])        #对系列中的每个元素应用函数
filter(func,seq)                 #返回其函数为真的元素的列表
reduce(func,seq[,inital])        #等同于func(func(func(seq[0],seq[1]),seq[2]).....)    二元操作函数    
sum(seq)                         #返回seq中所有元素的和
apply(func[,args[,kwargs]])      #调用函数，可以提供参数            参考p128


# In[5]:

#对象
#多态 （polymorphism）  程序得到一个对象，不知道他是怎么实现的 ，绑定到对象特性上面的函数称为方法（method）
#封装 （encapsulation）
#继承 （inheritance）
a=1,2,3
isinstance(a,str)    #isinstance 检查对象类型  tuple  list  dict  str  ,     type ,issubclass 
from random import choice     #choice  从系列中随机选择元素



# In[275]:

__metaclass__ = type  #确定使用新式类
class Person:
    def setName(self,name):         # self  自动将第一个参数传入函数中 ，没有他成员方法就没法访问他们要对其特性进行操作的对象本身
        self.name = name
    def getName(self):
        return self.name
    def greet(self):
        print("hello,world! i'm %s."%self.name)


foo=Person()
foo.setName('RaoXinLe')
foo.greet()


# In[168]:

#特性，函数方法   self 参数是方法与函数的区别，方法他将第一个参数绑定嗲所属的实例上，因此这个参数可以不提供，
class Class:
    def method(self):
        print("i have a self!")
def function():
    print("i don't...")


instance = Class()
instance.method()
instance.method=function()
instance.method


# In[169]:

class Bird:
    song = 'Squaawk'
    def sing(self):
        print(self.song)

bird=Bird()
bird.sing()


# In[171]:

#私有化
class Secretive:
    def __inaccessible(self):         # 为了防方法或特性变为私有（从外部无法访问），只要在他的名字前加双下划线即可： __     但是如果确实要访问      
        print("Bet you can't see me...")          #则可以    Secretive._Secretive__inaccessible()   以这种形式访问
    def accessible(self):
        print("the secret message is:")
        self.__inaccessible()

s=Secretive()
s.accessible()
s._Secretive__inaccessible()


# In[68]:

class membercounter: 
    members=0       # 特性
    def init(self): #方法
        membercounter.members += 1          #可以使用   self.members  ,也能用    membercounter.members
        #self.members += 1
        #return membercounter.members
        #return self.members

m1=membercounter()
m1.init()
m1.members        


# In[279]:

#指定超类 
class Filter:
    def init(self):
        self.blocked =[]
    def filter(self,sequnece):
        return [ x for x in sequence if x not in self.blocked]

class APAMFilter(Filter):     # 指定超类   
    def init(self):
        self.clocked = ['APAM']


# In[174]:

issubclass(APAMFilter,Filter)   #查看一个类是否是另一个类的子类   
APAMFilter.__bases__            #查看基类
s=APAMFilter()
isinstance(s,APAMFilter)        #检查某一对象是否是一个类的实例
s.__class__                     #检查对象属于某一类


# In[178]:

#多态和方法
from random import choice
x = choice(['hello,world!',[1,2,'e','e',4]])
x.count('e')                                     # 不用关心是列表还是字符串


# In[278]:

#repr(MyNames)      #str()一般是将数值转成字符串。 
                   #repr()是将一个对象转成字符串显示，注意只是显示用，有些对象转成字符串没有直接的意思。如list,dict使用str()是无效的，但使用repr可以，这是为了看它们都有哪些值，为了显示之用。 
    
def length_message(x):
    print("the length of" ,repr(x), "is" ,len(x))
    
    
length_message('MyNames')


# In[186]:

foo=lambda x: x*x      #lambda 表达式  
foo(2)
def foo1(x): return x*x
foo1(2)
#issubcalss(a,b)   #查看一个类是否是另一个类的子类   


# In[187]:

#接口和内省
#hasattr(Filter,'filter')      #检查方法是否在类中存在
getattr(Filter,'filter',None)  #获取对象
setattr(Filter,'name','xlrao'，None) #设置对象的特性
getattr(Filter,'name',None)    #获取对象的特性
Filter.__dict__                #对象内存储的所有值
#mappingproxy({'filter': <function Filter.filter at 0x7fa82c4910d0>, '__weakref__': <attribute '__weakref__' of 'Filter' objects>, 'name': 'xlrao', '__doc__': None, 'init': <function Filter.init at 0x7fa82c491158>, '__module__': '__main__', '__dict__': <attribute '__dict__' of 'Filter' objects>})


# In[293]:

dir(Exception)

Exception.__dict__


# In[95]:

#异常处理        ,每个异常都是一些类的实例         P146
#1/0              #，ZeroDivisionError
#raise语句         #raise ZeroDivisionError('The zero is not allow')  
#raise Exception('hyperdrive overload')   #引发异常类
#issubclass(ZeroDivisionError,Exception)
#Exception                                #所有异常类的基类
raise hyperdrive overload
dir(Exception)
calss SomeCustomeException(Exception): pass       #所有异常类的基类都是 Exception


#Exception                        所有异常的基类
#AttributeError                 特性应用或赋值失败时引发
#IOError                             试图打开不存在的文件时引发
#IndexError                       在使用序列中不存在的索引时引发
#KeyError                          在使用映射不存在的键时引发
#NameError                       在找不到名字（变量）时引发
#SyntaxError                     在代码为错误形式时引发
#TypeError                         在内建操作或者函数应用于错误类型的对象是引发
#ValueError                       在内建操作或者函数应用于正确类型的对象，但是该对象使用不合适的值时引发
#ZeroDivisionError          在除法或者摸除操作的第二个参数为0时引发

#warnings.filterwarnings(action....)    用于过滤警告


# In[189]:

#捕捉异常
try:
    x = input('enter the first number: ')
    y = input('enter the second number: ')
    print(int(x)/int(y))
except ZeroDivisionError:
    print("the second number can't be zero")




# In[190]:

#关闭及打开异常传递
class MuffledCalculator:
    muffled = False
    def calc(self,expr):
        try:
            return eval(expr)
        except ZeroDivisionError:               #也能用一个except(ZeroDivisionError,TypeError) 捕获多个异常   也能使用多个 except
            if  self.muffled:
                print("Division by zero is illegal")
            else:
                raise                                  #依然要传递异常

a=MuffledCalculator()
a.muffled = True   #  a.muffled = False  则屏蔽异常
a.calc('5/0')


# In[127]:

#捕捉异常对象
while True:                      #代码只没有发生异常时才会退出，异常发生会不断要求重新输入
    try:
        x = input('enter the first number: ')
        y = input('enter the second number: ')
        print(int(x)/int(y))
    except (ZeroDivisionError,TypeError,ValueError) as e:#将异常保留下来  ， 2.0的写法是 except (ZeroDivisionError,TypeError,ValueError) , e:
    #except (Exception) as e:    #可以捕捉更多的异常
        print(e)
        print("please try again!")
    else:
        break
    finally:
        print("cleaning up.")
        del x, y 


# In[128]:

def division(x,y):
    if y == 0 :
        raise ZeroDivisionError('The zero is not allow')
    return x/y  
division(10,1)


# In[195]:

#方法，属性，迭代器      在python中  __xxx__  由这些名字组成的集合所包含的方法称为魔法方法
#__metaclass__=type     
#构造方法
class Foobar:
    def __init__(self,value = 42):      #构造方法
        self.somevar = value
        

f=Foobar('BC')
f.somevar

#__del__   #需要尽量避免使用该方法

__len__(self)          #返回集合所含的项目数
__getitem__(self,key)  #返回与所给键对应的值
__setitem__(self,key,value)  
__delitem(self,key)


# In[ ]:




# In[301]:

#魔术方法
def checkIndex(key):                #p163
    ''' xxxxxx'''
    if not isinstance(key,(int)):raise TypeError
    if key < 0: raise IndexError

class ArithmeticSequence:
    def __init__(self,start=0,step=1):
        '''xxxxxx'''
        print("1111")
        self.start=start
        self.step=step
        self.changed={}
    def __getitem__(self,key):
        print("2222")
        checkIndex(key)
        try:
            return self.changed[key]
        except KeyError :
            #print(e)
            return self.start + key*self.step
    def __setitem__(self,key,value):
        print("3333")
        checkIndex(key)
        self.changed[key]=value
    def __len__(self):
        print("4444")
        return len(self.changed)


# In[312]:

s=ArithmeticSequence(1,2)
s.__setitem__(1,3)


# In[218]:

len(s)


# In[314]:

#静态方法和类成员方法   Staticmethod   Classmethod   ,静态方法定义没有self参数，  类方法定义时需要Cls的类似 self参数    ???
__metaclass__=type
class MyClass:
    @staticmethod
    def smeth():      #不需要声明可以直接调用    MyClass.smeth()  
        print("this is a static method")
    #smeth = staticmethod(smeth)
    @classmethod
    def cmeth(cls):  #不需要声明可以直接调用    MyClass.cmeth()  
        print("this is a class method of " ,cls)
    #cmeth = classmethod(cmeth)


# In[315]:

MyClass.smeth()
MyClass.cmeth()


# In[223]:

#property  创建一个属性，其中访问器函数被用作参数（先取值后赋值） ，这个属性命名为size 
__metaclass__=type
class Rectangle:
    def __init__(self):
        self.width=0
        self.height=0
    def setSize(self,size):
        self.width,self.height=size
    def getSize(self):
        return self.width,self.height
    size = property(getSize,setSize)
    
#r=Rectangle()
#r.width=10
#r.height=5
#r.size
#r.size=150,100
#r.width


# In[316]:

r=Rectangle()
r.width=10
r.height=5
r.size
r.size=150,100
r.width


# In[232]:

# __getattr__(self,name),   当特性name被访问且对象没有相应的特性是被自动调用
# __getattribute__(self,name),    
# __setattr__(self,name,value),
# __delattr__(self,name)

class Rectangle:
    def __init__(self):
        self.width=10
        self.height=15
    def __setattr__(self,name,value):
        if name=='size':
            self.width,self.height=value
        else:
            self.__dict__[name]=value
    def __getattr__(self,name):
        if name=='size':
            return self.width,self.height
        else:
            raise AttributeError
            


# In[230]:

#迭代器

#__iter__  该方法返回一个迭代器  ，迭代器就是具有next方法的对象  ，如果next被调用但迭代器没有值可返回就会引发一个StopIteration异常
#3.0中迭代器对象应该事先__next__方法，而不是next，next（it)  等同于3.0之前版本中的it.next()
#迭代器与列表的区别   迭代器是计算一个值获取一个值，而列表是一次性获取所有值，如果值很多就会占用太多内存


class Fibs:
    def __init__(self,max): #构造函数 ，初始化
        self.max=max
        self.a=0
        self.b=1
        self.c=0
        print('init')
    def __iter__(self):    #生成迭代器
        print('iter')
        return self
    
    def __next__(self):    #迭代方法   def __next__(self):    
        print("Iterate %s" % self.c)
        if self.c > self.max:
            raise StopIteration()
        else:
            self.c += 1
        self.a,self.b =self.b,self.a+self.b
        return self.a

fibs=Fibs(5)
for f in fibs:
    if f>10:
        print(f)
        break
        
#list(fibs)     # 从迭代器得到系列。     


# In[40]:

#iter 可以从迭代的对象中获取迭代器
it=iter([1,2,3])
next(it)         #python 2   it.next()
next(it)

for f in it:
    if f>2:
        print(f)
        break


# In[231]:

#从迭代器得到系列
class TestIteratior:
    value = 0
    def __iter__(self):   
        return self
    def __next__(self):
        self.value += 1
        if self.value > 10:
            raise StopIteration
        else:
            return self.value
ti=TestIteratior()
list(ti)
    


# In[232]:

#生成器  用普通函数语法定义的迭代器
nested=[[1,2],[3,4],[5]]
def faltten(nested):
    for sublist in nested:
        for element in sublist:
            yield element             #任何包含yield语句的函数成为生成器  ， 函数会被冻结在yield处直至下次被激活，并从次点开始继续执行下一步

#调用
for i in faltten(nested):
    print(i)


# In[233]:

#递归生成器 （recursion），处理任意层嵌套  ,不要迭代字符串   send p174
def flatten(nested):
    try:
        for sublist in nested:
            for element in flatten(nested):  #递归
                yield element
    except TypeError:
        yield nested


# In[320]:

#模块  ，模块是程序，任何python 程序度可以作为模块导入
#hello.py   , c:\python
#print "hello,world!"
#将程序所在路径加入到默认路径中
import sys 
#sys.path.append("c:/python")  #在unix中则需要使用完整的路径   /home/xlrao/python    , sys.path.expanduser('~/python')

#import hello        #导入模块   并生成新的文件 .pyc
#hello,world!
#模块在第一次导入到程序中时别执行


sys.version_info


# In[ ]:




# In[ ]:

#带有测试代码的简单模块
#hello3.py----------------
def hello():
    print("hello,world!")
#A test:
hello()      #测试代码
#-------------------------
import hello3   #导入模块
hello,world!    #测试代码被执行了   ，？  如果不想执行测试代码怎么处理？  
hello3.hello()  #调用模块看书
hello,world!    #输出

#hello4.py----------------
def hello():
    print("hello,world!")
def test:
    hello()      #测试代码
if __name__ == '__main__': test()      #主程序中的变量__name__='__mian__' ,而在导入模块中，该变量的值为模块名  __name__='hello4'
#-------------------------
import hello4   #导入模块
hello3.hello()  #调用模块看书
hello,world!    #输出


# In[169]:

#模块放置位置
import sys,pprint        #pprint.pprint 相比print 能够提供更智能的打印输出
pprint.pprint(sys.path)
#命名模块  包含模块代码的文件的名字要和模块名一样，再加上.py扩展名， 在window系统中你也可以使用.pyw 
#包  为了组织好模块，可以将它们分组为包(package)，,包就是模块所在的目录， 他必须包含一个命名为__init__.py的文件（模块） p189
#文件constants/__init__.py 包括PI=3.14
#import constants
#print constants.PI

#查看模块包含的内容可以用   dir()函数，他会将对象（模块的函数，类，变量等）的所有特性列出，

import copy 
dir(copy)
[n for n in dir(copy) if not n.startswith('_')]


# In[175]:

#__all__变量   ,  __all__是在模块中设置的， 如__all__=["error",""copy,""deepcopy]  用于定义模块公有接口（public interface），告诉模块导入所有名字代表什么含义
#因此你使用如下代码  from copy import *  ，只能使用__all__变量中的4个要素 ，其他要素则要显示导入  如 from copy import PyStringMap
dir(copy)
copy.__all__ 
#__file__     #源代码 ，
print(copy.__file__)     #返回文件路径 /opt/anaconda3/lib/python3.5/copy.py
f=open('/opt/anaconda3/lib/python3.5/copy.py')
import pprint
pprint.pprint(f.readlines())


# In[183]:

#标准库
#sys   
#os
#fileinput    p179
#集合    
set(range(10))
#堆（heap）  p200
#time        p202
#random
#shlve
#re



# In[197]:

f=open('somefile.txt')
f.readlines()
f.close()
import fileinput       #
fileinput.filename()   #返回正在处理的文件名
fileinput.lineno()     #返回当前行的行数
fileinput.filelineno() #不同文件会重置行号 

fileinput.nextfile()
fileinput.input()
f=open('somefile.txt')


# In[201]:

import fileinput  
f=open('somefile.txt')
for line in fileinput.input(inplace=True):
    line=line.rstrip()
    num=fileinput.lineno()
    print('%-40s # %2i'%(line,num))
    


# In[321]:

sys.argv


# In[236]:

#time  日期时间   ： 获得当前时间，操作时间，从字符串读取时间，以及格式化时间为字符串     
#日期元组  
#d=(2008，1，21,12,2,56,0,21,0)
#d[0]=2008    年
#d[1]=1       月
#d[2]=21      日
#d[3]=12      时
#d[4]=2       分
#d[5]=56      秒
#d[6]=0       周     （周一为0  0~6）
#d[7]=21      儒历日 （1~336）
#d[8]=0       夏令时  （0,1，-1） 布尔   如果为-1，mkttime该函数将此元组转换成时间戳， 从19700101开始一秒计量
d=(2008,1,21,12,2,56,0,21,0)
import time 
time.asctime()   #'Mon Sep  5 16:15:40 2016'  将当前时间转化成格式化字符串
time.localtime() #time.struct_time(tm_year=2016, tm_mon=9, tm_mday=5, tm_hour=16, tm_min=17, tm_sec=4, tm_wday=0, tm_yday=249, tm_isdst=0)
time.gmtime()    #全球同一时间  
time.mktime(d)   #  1200888176.0  与localtime功能相反  
time.sleep(1)    #让解释器等待给定的秒数
time.strptime(time.asctime())  #将asctime转成日期元组
time.time()


# In[239]:

time.gmtime()


# In[234]:

#random    返回随机数
import random
random.__all__



# In[235]:

#re  正则表达式
# .  匹配任何字符  ，不含换行符
# \\ 对特殊字符进行转义
# '[a-zA-Z0-9]'  字符集  ， [^a-z] 反转字符集
#'python|perl 选择符   'p(ython|erl’
# ?  可选项和重复子模式  可出现也可不出现
# *  允许0次货多次
# +  允许1次或多次
# {m,n}允许重复m~n次

import re         #p213
re.__all__

#if re.search(pat,string):
#    print('fount it')


# In[90]:

#文件和素材
f=open('/opt/anaconda3/lib/python3.5/copy.py')


# In[80]:

[n for n in dir(copy) if not n.startswith('_')]


# In[ ]:

#文件和素材
#打开文件
open(name[, mode[,buffering]])
f=open(r'\text\somefile.txt')    #如果文件不存在则会报IOError   , f=open(r+'\text\somefile.txt')  ,'+' 表示读写读允许
#r 读，w 写,a 追加，b 二进制模式  + 读写
#缓冲 0无缓冲，1有缓冲,当有缓冲时只有使用flush或close 才会更新硬盘上的数据，大于1的数字表示缓冲的大小（字节），-1或者任何负数表示默认大小


f=open(r'\text\somefile.txt','w')
f.wirte('hello,')
f.write('world')
f.close()

f=open(r'\text\somefile.txt','r')
f.read(4)
'hell'
f.read()
o,world

#基本文件方法
f=url.urlopen()   #打开网页内容
read,readline,readlines    

#三种标准流
#sys 模块
#数据输入标准 sys.stdin  ,当程序从标准输入读取数据时，可以通过输入或者使用管道把他和其它程序的标准输出链接起来，提供文本。要打印的文本保存在sys.stdout中，
#该数据一般显示在屏幕上，但也能通过管道链接到其它程序的输入
#错误信息被写入sys.stderr


# In[ ]:

#管式输出
#在unix的shell中，使用管道可以在一个命令后面续写其他的多个命令，
$cat somefile.txt|python somescript.py| sort
$cat somefile.txt 把somefile的内容写到标准输出 sys.stdout
python somescript.py运行脚本， 从标准输入读，并写入到标准输出中
sort  从标准输出读取所有文本

'|'管道符号的作用，管道符号将一个命令的输出和下一命令的输入连在一起

#somescript.py
import sys
text=sys.stdin.read()  #从标准输入中读取
words=text.split()
wordcount=len(words)
print('wordcount:',cordcount)

cat somefile.txt|python somescript.py   输出结果  wordcount:11

try:
    #
finally:
    file.close()   #关闭文件
    
from __future__ import with_statement  
with语句打开文件并赋值到变量上，with open("somefile.txt") as somefile:
    do_something(somefile)
    


# In[98]:

#使用基本文件方法
# !pwd  #获取当前路径 /opt/workspace/ipython_ws/xlrao  

f=open('somefile.txt')
f.read(7)
f.read(4)
print(f.read())
#f.close()


# In[101]:

f=open('somefile.txt')
for i in range(3):
    print(str(i) +':'+ f.readline())     # str(i) 将数字转化成字符串

import pprint  
pprint.pprint(open( 'somefile.txt').readlines())   #open( 'somefile.txt').readlines() 返回列表


f=open('somefile.txt')
lines=f.readlines()
f.close
lines[1]="isn't a\n"
f=open('somefile.txt','w')
f.writelines(lines)
f.close()


# In[107]:

#写文件
f=open('somefile.txt')
lines=f.readlines()
f.close
lines[1]="isn't a\n"
f=open('somefile.txt','w')
f.writelines(lines)
f.close()

f=open('somefile.txt')
f.read()


# In[111]:

#对文件内容进行迭代
def process(string):
    print('processing:',string)

f=open('somefile.txt')
char=f.read(1)
while(char):
    process(char)
    char=f.read(3)
f.close()



# In[119]:

import fileinput
for line in fileinput.input('somefile.txt'):
    process(line)
    
#截至到11章

#import sys
#for line in sys.stdin:
#    process(line)


# In[163]:

#配置文件  config.py   ， 在此文件中可以配置一些进场使用的变量， 然后在使用时引入该模块    p334
#congif.py  配置文件名
#pi=3.1415926
#greeting='welcome to the area calculation program'
#question='please enter the radius:'
#result_message='the area is:'

from config import pi
pi


# In[162]:

#另一种方法 ConfigParser  配置文件格式
#config.txt  配置文件名
#[numbers]
#pi: 3.1415926535897931
#[messages]
#greeting: welcome to the area calculation program
   
import configparser
CONFIGFILE="config.txt"
config=configparser.ConfigParser()      
#读取配置文件：
config.read(CONFIGFILE)
config.get('messages','greeting')
radius=input(config.get('messages','greeting')+' ')


# In[157]:

config.read('config.txt')
config.get('numbers','pi')


# In[165]:

configparser.__all__


# In[ ]:






