from sklearn.neural_network import MLPClassifier
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd     



#读取csv文件
www =pd.read_csv('www_62145.csv',header=None)
mail =pd.read_csv('mail_28567.csv',header=None)
bulk =pd.read_csv('bulk_11539.csv',header=None)
database=pd.read_csv('database_2648.csv',header=None)
services=pd.read_csv('services_2099.csv',header=None)
p2p=pd.read_csv('P2P_2094.csv',header=None)

    
def run(i):
    #读取其中的x
    x_www = www.values[0:dataNum[0][i],:]    
    x_mail = mail.values[0:dataNum[1][i],:] 
    x_bulk = bulk.values[0:dataNum[2][i],:]    
    x_database = database.values[0:dataNum[3][i],:] 
    x_services = services.values[0:dataNum[4][i],:]  
    x_p2p = p2p.values[0:dataNum[5][i],:]  

    x_www = preprocessing.normalize(x_www)
    x_mail = preprocessing.normalize(x_mail)
    x_bulk = preprocessing.normalize(x_bulk)
    x_database = preprocessing.normalize(x_database)
    x_services = preprocessing.normalize(x_services)
    x_p2p = preprocessing.normalize(x_p2p)
    
    serie = pd.Series([dataNum[0][i],dataNum[1][i],dataNum[2][i],dataNum[3][i],dataNum[4][i],dataNum[5][i]],index =['a0','a1','a2','a3','a4','a5'])
    serie05 = serie.sort_values(ascending=False)
    serieList = serie05.index.tolist()
    a0=serieList.index('a0')
    a1=serieList.index('a1')
    a2=serieList.index('a2')
    a3=serieList.index('a3')
    a4=serieList.index('a4')
    a5=serieList.index('a5')
   
    #读取其中的y
    y_www=np.array([a0]*len(x_www))   
    y_mail =np.array([a1]*len(x_mail))
    y_bulk =np.array([a2]*len(x_bulk))    
    y_database= np.array([a3]*len(x_database))    
    y_services =np.array([a4]*len(x_services))
    y_p2p=np.array([a5]*len(x_p2p))
   
    #分成训练集和测试集
    x_tr_www,x_te_www,y_tr_www,y_te_www = train_test_split(x_www,y_www,test_size=0.3,random_state=1) 
    x_tr_mail,x_te_mail,y_tr_mail,y_te_mail = train_test_split(x_mail,y_mail,test_size=0.3,random_state=1)  
    x_tr_bulk,x_te_bulk,y_tr_bulk,y_te_bulk = train_test_split(x_bulk,y_bulk,test_size=0.3,random_state=1)  
    x_tr_database,x_te_database,y_tr_database,y_te_database = train_test_split(x_database,y_database,test_size=0.3,random_state=1)  
    x_tr_services,x_te_services,y_tr_services,y_te_services = train_test_split(x_services,y_services,test_size=0.3,random_state=1)    
    x_tr_p2p,x_te_p2p,y_tr_p2p,y_te_p2p = train_test_split(x_p2p,y_p2p,test_size=0.3,random_state=1)
         
    #制作训练集
    x_train = np.concatenate((x_tr_www,x_tr_mail,x_tr_bulk,x_tr_database,x_tr_services,x_tr_p2p))
                   
    y_train1 = np.concatenate((y_tr_www,y_tr_mail,y_tr_bulk,y_tr_database,y_tr_services,y_tr_p2p))
    y_train = np.reshape(y_train1, (y_train1.shape[0],1))
    
    #制作测试集
    x_test = np.concatenate((x_te_www,x_te_mail,x_te_bulk,x_te_database,
                              x_te_services,x_te_p2p))
    y_test1 = np.concatenate((y_te_www,y_te_mail,y_te_bulk,y_te_database,
                              y_te_services,y_te_p2p))
                        
    y_test = np.reshape(y_test1,(y_test1.shape[0],1))
    
    #训练集打散    
    training_data1 = np.hstack([x_train, y_train])
    np.random.shuffle(training_data1)
    X_tr = training_data1[:,:-1]   
    y_tr1 = training_data1[:,-1]
    y_tr = np.reshape(y_tr1,(y_tr1.shape[0],1))

    #测试集打散 
    global y_te
    global predict_results
    test_data1 = np.hstack([x_test, y_test])
    np.random.shuffle(test_data1)
    X_te = test_data1[:,:-1] #X_te = test_data1[:,:-1]
    y_te1 = test_data1[:,-1]
    y_te =np.reshape(y_te1,(y_te1.shape[0],1))
    #分类器
    clf =  MLPClassifier(hidden_layer_sizes=(30,20,10),max_iter=50,batch_size=50)
    clf.fit(X_tr,y_tr)
    predict_results=clf.predict(X_te)
    
def write1():
    print("accuracy:")
    accuracy_data = accuracy_score(y_te,predict_results)
    accuracy_list = []
    accuracy_list.append(accuracy_data)
    with open("01accuracy.csv","a+", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows([accuracy_list])


    print("precision值:")
    precision_data = precision_score(y_te, predict_results,average=None)
    precision_list = precision_data.tolist()
    with open("01precision.csv","a+", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows([precision_list])
    print(precision_data)

    
    print("recall值:")
    recall_data = recall_score(y_te, predict_results,average=None)
    recall_list = recall_data.tolist()
    with open("01recall.csv","a+", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows([recall_list])
    print(recall_data)

    
    print("f1值:")
    f1_data = f1_score(y_te, predict_results,average=None)
    f1_list = f1_data.tolist()
    with open("01f1.csv","a+", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows([f1_list])
    print(f1_data)

    
def write2():
    print("accuracy:")
    accuracy_data = accuracy_score(y_te,predict_results)
    accuracy_list = []
    accuracy_list.append(accuracy_data)
    with open("02accuracy.csv","a+", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows([accuracy_list])

    print("precision值:")
    precision_data = precision_score(y_te, predict_results,average=None)
    precision_list = precision_data.tolist()
    with open("02precision.csv","a+", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows([precision_list])
    print(precision_data)
    
    print("recall值:")
    recall_data = recall_score(y_te, predict_results,average=None)
    recall_list = recall_data.tolist()
    with open("02recall.csv","a+", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows([recall_list])
    print(recall_data)
    
    print("f1值:")
    f1_data = f1_score(y_te, predict_results,average=None)
    f1_list = f1_data.tolist()
    with open("02f1.csv","a+", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows([f1_list])
    print(f1_data)
    
def write3():
    print("accuracy:")
    accuracy_data = accuracy_score(y_te,predict_results)
    accuracy_list = []
    accuracy_list.append(accuracy_data)
    with open("03accuracy.csv","a+", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows([accuracy_list])

    print("precision值:")
    precision_data = precision_score(y_te, predict_results,average=None)
    precision_list = precision_data.tolist()
    with open("03precision.csv","a+", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows([precision_list])
    print(precision_data)
    
    print("recall值:")
    recall_data = recall_score(y_te, predict_results,average=None)
    recall_list = recall_data.tolist()
    with open("03recall.csv","a+", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows([recall_list])
    print(recall_data)
    
    print("f1值:")
    f1_data = f1_score(y_te, predict_results,average=None)
    f1_list = f1_data.tolist()
    with open("03f1.csv","a+", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows([f1_list])
    print(f1_data)    
    
def write4():
    print("accuracy:")
    accuracy_data = accuracy_score(y_te,predict_results)
    accuracy_list = []
    accuracy_list.append(accuracy_data)
    with open("04accuracy.csv","a+", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows([accuracy_list])

    print("precision值:")
    precision_data = precision_score(y_te, predict_results,average=None)
    precision_list = precision_data.tolist()
    with open("04precision.csv","a+", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows([precision_list])
    print(precision_data)
    
    print("recall值:")
    recall_data = recall_score(y_te, predict_results,average=None)
    recall_list = recall_data.tolist()
    with open("04recall.csv","a+", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows([recall_list])
    print(recall_data)
    
    print("f1值:")
    f1_data = f1_score(y_te, predict_results,average=None)
    f1_list = f1_data.tolist()
    with open("04f1.csv","a+", newline="") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows([f1_list])
    print(f1_data)       

    
    
if __name__ == '__main__':
    dataNum = pd.read_csv('类型4\\a0=0.002\数据设置.csv',header=None)
    for i in range(18):
        for j in range(20):
            print('第' + str(j) + '次')
            run(i)  
            write1()


if __name__ == '__main__':
    dataNum = pd.read_csv('类型4\\a1=0.01\数据设置.csv',header=None)
    for i in range(18):
        for j in range(20):
            print('第' + str(j) + '次')
            run(i)  
            write2()
            
if __name__ == '__main__':
    dataNum = pd.read_csv('类型4\\a2=0.02\数据设置.csv',header=None)
    for i in range(18):
        for j in range(20):
            print('第' + str(j) + '次')
            run(i)  
            write3()
            
if __name__ == '__main__':
    dataNum = pd.read_csv('类型4\\a3=0.1\数据设置.csv',header=None)
    for i in range(18):
        for j in range(20):
            print('第' + str(j) + '次')
            run(i)  
            write4()


