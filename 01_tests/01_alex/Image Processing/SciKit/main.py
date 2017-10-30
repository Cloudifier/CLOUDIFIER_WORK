import pandas as pd
import numpy as np
from sklearn import datasets, svm
from os import listdir
from os.path import isfile, join


dataPath = r'D:\Personal\Projects\Cloudifier\Poze\Processed';
trainingDataPercent = 70;
testDataPercent = 30;
clf = svm.SVC(gamma=0.001, C=100.)
files = [f for f in listdir(dataPath) if isfile(join(dataPath, f))]
maxTrainingFileIndex = int(round(len(files) * trainingDataPercent / 100))
#training area
numberOfTrainingFiles = 0
for file in files[:maxTrainingFileIndex]:
    numberOfTrainingFiles = numberOfTrainingFiles + 1
    path = dataPath + '\\' + file;
    df = pd.read_csv(path, header=0)
    datas = df[df.columns[2:]].values
    nans = np.isnan(datas)
    datas[nans] = 0
    labels = df[df.columns[1]].values
    clf.fit(datas, labels)
print('##############TESTING##############')
#testing area
numberOfTestingFiles = 0
for file in files[maxTrainingFileIndex:]:
    numberOfTestingFiles = numberOfTestingFiles + 1
    path = dataPath + '\\' + file;
    df = pd.read_csv(path, header=0)
    datas = df[df.columns[2:]].values
    nans = np.isnan(datas)
    datas[nans] = 0
    predict = clf.predict(datas)
    print(predict)
    print('-----------------------')

print('Training Files: ' + numberOfTrainingFiles)
print('Testing Files: ' + numberOfTestingFiles)
#
# for file in files[:maxTrainingFileIndex]:
#     path = dataPath + '\\' + file;
#     df = pd.read_csv(path, header=0)
#     datas = df[df.columns[2:]].values
#     nans = np.isnan(datas)
#     datas[nans] = 0
#     labels = df[df.columns[1]].values
#     clf.fit(datas, labels)
# print('##############TESTING##############')
# #testing area
# for file in files[maxTrainingFileIndex:]:
#     path = dataPath + '\\' + file;
#     df = pd.read_csv(path, header=0)
#     datas = df[df.columns[2:]].values
#     print(datas)
#     nans = np.isnan(datas)
#     datas[nans] = 0
#     predict = clf.predict(datas)
#     print(predict)
#     print('-----------------------')



# #remember that you tried to pass entire dataset (datas) to the fit with the corresponding labels but you got the error bellow
# #ValueError: The number of classes has to be greater than one; got 1
# #as far as you understood at that time, the labels array has to contain at least two different values inside. we only have 'button' all over the place
# #in order to avoid this error, we will iterate through the dataframe and pass each row to the fit
# clf = svm.SVC(gamma=0.001, C=100.)
# # for index, row in datas.iterrows(): #iterate dataframe to avoid exception
#     # clf.fit(row, labels[index])
#     # print(row)
# i = 0
#
# # for row in datas:
# #     print(row)
#     # clf.fit(row, 3)
#     # i = i + 1