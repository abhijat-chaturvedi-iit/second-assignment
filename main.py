from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import sklearn
from skimage.transform import rescale, resize
from skimage import transform
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn import tree

Gamma_list=[0.01 ,0.001, 0.0001, 0.0005]
c_list=[.1 ,.5, .4, 10, 5, 1]

h_param_comb=[{'gamma':g,'C':c} for g in Gamma_list for c in c_list]

digits = datasets.load_digits()



import numpy as np 
def resize_a(image,n):
    image = resize(image, (image.shape[0] // n, image.shape[1] // n),anti_aliasing=True)
    return image
def resize_b(image,n):
    image = resize(image, (image.shape[0]*n, image.shape[1]*n),anti_aliasing=True)
    return image

digits_4 = np.zeros((1797, 2, 2))  
digits_2 = np.zeros((1797, 4, 4)) 
digits_5 = np.zeros((1797, 16, 16))  

for i in range(0,1797):
    digits_4[i] = resize_a(digits.images[i],4)

for i in range(0,1797):
    digits_2[i] = resize_a(digits.images[i],2)

for i in range(0,1797):
    digits_5[i] = resize_b(digits.images[i],2)
    
      
n_samples = len(digits_5)
data = digits_5.reshape((n_samples, -1))
print('\n')
# print(digits_5[-1].shape)
print("Image size in dataset",digits.images[-1].shape)

compare_list_svm = []
# train_frac=0.8
# test_frac=0.1
# dev_frac=0.1
split_list =  [[0.8,0.1,0.1],[0.70,0.15,0.15],[0.6,0.2,0.2],[0.50,0.25,0.25],[0.4,0.3,0.3]]
for each_split in split_list:
    print(f"Running for {each_split}\n")
    train_frac = each_split[0]
    test_frac = each_split[1]
    dev_frac = each_split[2]
    dev_test_frac=1-train_frac

    X_train, X_dev_test, y_train, y_dev_test = train_test_split(data ,digits.target, test_size=dev_test_frac, shuffle=True,random_state=42)
    X_test, X_dev, y_test, y_dev = train_test_split(X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True,random_state=42)

    best_acc=-1
    best_model=None
    best_h_params=None 
    df = pd.DataFrame(columns=['parameters','train', 'dev', 'test'])

    for com_hyper in h_param_comb:

        clf = svm.SVC()

        hyper_params=com_hyper
        clf.set_params(**hyper_params)

        clf.fit(X_train, y_train)

        predicted_train = clf.predict(X_train)
        predicted_dev = clf.predict(X_dev)
        predicted_test = clf.predict(X_test)
        
        
        cur_acc_train=metrics.accuracy_score(y_pred=predicted_train,y_true=y_train)
        cur_acc_dev=metrics.accuracy_score(y_pred=predicted_dev,y_true=y_dev)
        cur_acc_test=metrics.accuracy_score(y_pred=predicted_test,y_true=y_test)
        
        if cur_acc_dev>best_acc:
            best_acc=cur_acc_dev
            best_model=clf
            best_h_params=com_hyper
            df2 = {'parameters': hyper_params,'train': str(cur_acc_train), 'dev': str(cur_acc_dev), 'test': str(cur_acc_test)}
            df = df.append(df2, ignore_index = True)
            # print("found new best acc with: "+str(com_hyper))
            # print("New best accuracy:"+ " train" + "  "+str(cur_acc_train)+ " "+ "dev" + " "+str(cur_acc_dev)+ " "+ "test" + " " +str(cur_acc_test))
            

            
    predicted = best_model.predict(X_test)
    print(df)
    print ('\n')
    print("Best hyperparameters were: ")
    print(com_hyper)
    print ('\n')
    print("Best accuracy on dev: ")
    compare_list_svm.append(best_acc)
    print(best_acc)


# train_frac=0.8
# test_frac=0.1
# dev_frac=0.1
compare_list_dt = []
split_list =  [[0.8,0.1,0.1],[0.70,0.15,0.15],[0.6,0.2,0.2],[0.50,0.25,0.25],[0.4,0.3,0.3]]
for each_split in split_list:
    print(f"Running for {each_split}\n")
    train_frac = each_split[0]
    test_frac = each_split[1]
    dev_frac = each_split[2]
    

    dev_test_frac=1-train_frac

    X_train, X_dev_test, y_train, y_dev_test = train_test_split(data ,digits.target, test_size=dev_test_frac, shuffle=True,random_state=42)
    X_test, X_dev, y_test, y_dev = train_test_split(X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True,random_state=42)

    best_acc=-1
    best_model=None
    best_h_params=None 
    df_dt = pd.DataFrame(columns=['parameters','train', 'dev', 'test'])

    max_depth_list = [10,20,30,40]

    for each in max_depth_list:
        clf = tree.DecisionTreeClassifier(max_depth=each)
        clf.fit(X_train, y_train)

        predicted_train = clf.predict(X_train)
        predicted_dev = clf.predict(X_dev)
        predicted_test_dt = clf.predict(X_test)
        
        
        cur_acc_train=metrics.accuracy_score(y_pred=predicted_train,y_true=y_train)
        cur_acc_dev=metrics.accuracy_score(y_pred=predicted_dev,y_true=y_dev)
        cur_acc_test=metrics.accuracy_score(y_pred=predicted_test_dt,y_true=y_test)
        
        if cur_acc_dev>best_acc:
            best_acc=cur_acc_dev
            best_model=clf
            best_h_params=each
            df2_dt = {'parameters': each,'train': str(cur_acc_train), 'dev': str(cur_acc_dev), 'test': str(cur_acc_test)}
            df_dt = df_dt.append(df2_dt, ignore_index = True)
            # print("found new best acc with: "+str(each))
            # print("New best accuracy:"+ " train" + "  "+str(cur_acc_train)+ " "+ "dev" + " "+str(cur_acc_dev)+ " "+ "test" + " " +str(cur_acc_test))

    predicted = best_model.predict(X_test)
    print(df_dt)
    print ('\n')
    print("Best hyperparameters were: ")
    print(each)
    print ('\n')
    print("Best accuracy on dev: ")
    compare_list_dt.append(best_acc)
    print(best_acc)


mean_svm = np.mean(compare_list_svm)
mean_dt = np.mean(compare_list_dt)

std_svm = np.std(compare_list_svm)
std_dt = np.std(compare_list_dt)

print(f"Mean for SVM = {mean_svm}")
print(f"Mean for DT = {mean_dt}")
print(f"STD for SVM = {std_svm}")
print(f"STD for DT = {std_dt}")


def returnNotMatches(a, b):
    i=0
    incorrect = []
    while i<len(a):
        if a[i]!=b[i]:
            incorrect.append(a[i])
        i=i+1
    return incorrect

incorrect = returnNotMatches( predicted_test,predicted_test_dt )
print(len(incorrect))
