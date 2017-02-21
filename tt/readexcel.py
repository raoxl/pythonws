#coding=utf-8
#@author:xiaolin
#@file:readExcels.py
#@time:2016/11/11 11:02

import re
import os
import xlrd
import csv
import pandas as pd
from pandas import DataFrame
import datetime
import csv

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def readExcel(file,csvfile):
    '''
    :param file: excel 文件
    :return: data frame
    '''
    writer = csv.writer(csvfile)
    datadf=pd.read_excel(file,header=4)#从这一行开始读取数据
    datadf=datadf.drop(datadf.index[-4:-1])#舍弃掉后面几行
    datadf=datadf.ix[:,(2,4,10)]
    rows = datadf.values.tolist()
    writer.writerows(rows)

def getFiles(datapath,csvfile):
    '''
    :param datapath: 存放excel 的文件夹
    :param savepath: 保存结果的路径 包含了文件名和后缀，后缀为 TXT
    :return: 直接保存文件
    '''
    for root,dirs,item in os.walk(datapath):
        i=1
        for file in item:
            file2 = root +str('//')+file
            print "READING THE NUM",i,"FILE",datetime.datetime.now()
            readExcel(file2,csvfile)
            i+=1
    csvfile.close()
    print ("DONE! ")

if __name__ == '__main__':
    datapath="E:/alipaydata/test"
    savepath="E:/alipaydata/test.txt"
    csvfile = file(savepath, 'w+')
    starttime=datetime.datetime.now()
    getFiles(datapath,csvfile)
    endtime=datetime.datetime.now()
    print (endtime-starttime).seconds




'''
Sub csv2XLS()
Dim FilePath, MyFile, iPath As String
iPath = ThisWorkbook.Path
MyFile = Dir(iPath & "\*.xlsx")
If MyFile <> "" Then
Do
    On Error Resume Next
    If MyFile = ThisWorkbook.Name Then MyFile = Dir
    Workbooks.Open (iPath & "\" & MyFile)
    MyFile = Replace(MyFile, ".xlsx", ".csv")
    Name = "\" & MyFile
    FilePath = iPath & Name
    Application.ScreenUpdating = False
    ActiveWorkbook.SaveAs Filename:=FilePath, FileFormat:=xlCSV, _
        CreateBackup:=False
    Workbooks(MyFile).Close True
    Application.ScreenUpdating = True
    MyFile = Dir
Loop While MyFile <> ""
End If
End Sub


'''