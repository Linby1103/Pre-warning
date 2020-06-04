from numpy import *
import matplotlib.pyplot as plt
import time
import numpy as np
from data import Dataset
import pickle
# calculate the sigmoid function
#算法推理来自于：https://blog.csdn.net/zouxy09/article/details/20319673
class LR(object):
    def __init__(self,train_data,label_):

        self.traindata=train_data
        self.label=label_

    def sigmoid(self,z):
        return 1.0 / (1 + exp(-z))

    def train(self,opts,weight_path="weights.pkl"):
        # train a logistic regression model using some optional optimize algorithm
        # traindata is a mat datatype, each row stands for one sample
        #label is mat datatype too, each row is the corresponding label
        #opts is optimize option include step and maximum number of iterations
        # calculate training time
        startTime = time.time()

        numSamples, numFeatures = shape(self.traindata)
        alpha = opts['alpha'];
        maxIter = opts['maxIter']
        weights = ones((numFeatures, 1))


        # optimize through gradient descent algorilthm
        for k in range(maxIter):
            if opts['optimizeType'] == 'gradDescent':  # gradient descent algorilthm
                output = self.sigmoid(self.traindata * weights)
                error = self.label - output  # loss MSE 1/2 * 1/len(train_y) *  (train_y_i-output_i)
                weights = weights + alpha * self.traindata.transpose() * error  # (y-err)*x  BP
            elif opts['optimizeType'] == 'stocGradDescent':  # stochastic gradient descent
                for i in range(numSamples):
                    output = self.sigmoid(self.traindata[i, :] * weights)
                    error = self.label[i, 0] - output
                    weights = weights + alpha * self.traindata[i, :].transpose() * error
            elif opts['optimizeType'] == 'smoothStocGradDescent':  # smooth stochastic gradient descent
                # randomly select samples to optimize for reducing cycle fluctuations
                dataIndex = range(numSamples)
                for i in range(numSamples):
                    alpha = 4.0 / (1.0 + k + i) + 0.01
                    randIndex = int(random.uniform(0, len(dataIndex)))
                    output = self.sigmoid(self.traindata[randIndex, :] * weights)
                    error = self.label[randIndex, 0] - output
                    weights = weights + alpha * self.traindata[randIndex, :].transpose() * error
                    del (dataIndex[randIndex])  # during one interation, delete the optimized sample
            else:
                raise NameError('Not support optimize method type!')

        print('training finished!  %fs!' % (time.time() - startTime))

        try:
            pickf = open(weight_path, 'wb')
            print("Save weight to {}".format(weight_path))
            data = {"weight": weights}
            pickle.dump(data, pickf)
        except Exception as e:
            print("Write weight to {} fail! {}".format(pickf,e))
            pickf.close()
        finally:
            pickf.close()

    # test  trained Logistic Regression model given test set
    def predict(self,weights, test_x, test_y):
        numSamples, numFeatures = shape(test_x)
        matchCount = 0
        result=[]
        for i in range(numSamples):
            prob = 1 if self.sigmoid(test_x[i, :] * weights)[0, 0] > 0.5 else 0
            result.append(prob)
            if prob == bool(test_y[i, 0]):
                matchCount += 1
        accuracy = float(matchCount) / numSamples
        print("accuracy={}".format(accuracy))
        return result
    def load_weight(self,path):
        weight={}
        try:
             weight = pickle.load(open(path,'rb'))["weight"]
             print(weight.shape)
        except Exception as e:
            print("Load weight faild! {}".format(e))
        finally:
            return weight



def loadData():
    train_x = []
    train_y = []
    fileIn = open('./data/testdata.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split()
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))
    return mat(train_x), mat(train_y).transpose()

if __name__=="__main__":
    STATUS = ['非ST','ST']
    save_model="weights.pkl"

    train, label = Dataset.GetDatafromDict()
    trainarrayx=mat(train)
    labelarray=mat(label).transpose()
    logistreg = LR(trainarrayx,labelarray)
    # print(trainarrayx.shape,labelarray.shape)
    opts={"alpha":0.001,"maxIter":200,"optimizeType":"gradDescent"}
    logistreg.train(opts,save_model)
    weights=logistreg.load_weight(save_model)
    res=logistreg.predict(weights,trainarrayx,labelarray)

    for i in range(len(res)):
        print("预测结果 :{}---原始数据:{}".format(STATUS[res[i]], STATUS[label[i]]))












