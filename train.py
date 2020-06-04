import numpy as np
import random
import joblib
from data import Dataset
from sklearn import linear_model
from sklearn.model_selection import train_test_split
SAMPLE_SIZE=1000


"""
Creat radom
"""
class CreateRandomPage:
    def __init__(self, begin, end, needcount):
        self.begin = begin
        self.end = end
        self.needcount = needcount
        self.resultlist = []
        self.count = 0
    def createrandompage(self):
        tempInt = random.uniform(self.begin, self.end)
        if(self.count < self.needcount):
            if(tempInt not in self.resultlist):
                self.resultlist.append(tempInt)    #将长生的随机数追加到列表中
                self.count += 1
            return self.createrandompage()      #在此用递归的思想
        return self.resultlist


def CreatTrainDataset():
    traindatas=[]
    labels=[]

    randompos=CreateRandomPage(0,100,2)# 单位：万元
    randomneg = CreateRandomPage(-100,0, 2)  # 单位：万元

    #Creat 1000 samples

    for _ in range(SAMPLE_SIZE):
         if random.randint(0, 10)%2 ==0 :
             traindatas.append(randompos.createrandompage())#正样本
             labels.append(1)
         else:
             traindatas.append(randomneg.createrandompage())#负样本
             labels.append(0)
    return np.array(traindatas),np.array(labels)


def Preprocess(TrainData,Label):#数据归一化
    TrainData = TrainData / 100
    # Label = Label / 100



    return TrainData,Label

def DataGenerator(TrainData,Label,batch_size):
    pass

def train():
    # trainx,trainy=CreatTrainDataset()
    trainx,trainy=Dataset.GetDatafromDict()
    X_train, X_test, Y_train, Y_test = train_test_split(trainx, trainy, test_size=0.3, random_state=0)

    #保持样本的随机性
    radom_sample=[ i for i in range (len(Y_train))]
    random.shuffle(radom_sample)
    X_train=X_train[radom_sample]
    Y_train=Y_train[radom_sample]


    nomtrainx, nomtrainy=Preprocess(X_train,Y_train)
    print(nomtrainx.shape,nomtrainy.shape)
    # 1、创建了一个线性逻辑分类器实例去拟合数据
    logreg = linear_model.LogisticRegression(C=1e5,max_iter=200)

    # 2、测试数据集
    test_data=np.expand_dims(trainx[2], axis=0)

    # 3、训练拟合函数
    logreg.fit( nomtrainx, nomtrainy)
    # 4、保存模型
    joblib.dump(filename='LR.model', value=logreg)
    # 5、预测结果
    out=logreg.predict(test_data)
    print(out)


def predict(modelpath):
    trainx, trainy = Dataset.GetDatafromDict()
    X_train, X_test, Y_train, Y_test = train_test_split(trainx, trainy, test_size=0.3, random_state=0)
    nomtrainx, nomtrainy = Preprocess(X_test, Y_test)
    model=joblib.load(filename=modelpath)

    # TestSample=len(Y_test)
    # possample=[ Y_train[i] for i in range(TestSample) if Y_test[i]==0]
    # negsample=[ Y_train[i] for i in range(TestSample) if Y_test[i]==1]

    TestSample = len(Y_test)
    possample=np.sum(Y_test)
    negsample=TestSample-possample


    res=model.predict(nomtrainx)

    TP,TN,FP,FN=0,0,0,0
    for id,i in enumerate(res):
        if res[i]==Y_test[id] and Y_test[id]==1:
            TP+=1
        elif res[i] == Y_test[id] and Y_test[id] == 0:
            TN+=1
        elif res[i] != Y_test[id] and Y_test[id] == 0:
            FP+=1
        else:
            FN+=1


        print("predict result :{}---label:{}".format(res[id],Y_test[id]))


    print("result:acc={} , recall={}".format((TP+TN)/TestSample,TP/(TP+FN)))




# 下面显示的是iris数据集中的逻辑回归分类器决策边界。数据点根据其标签进行着色。
if __name__ == "__main__":
    train()
    loadmodel='C:/Users/91324/Desktop/unet/LR.model'
    predict(loadmodel)