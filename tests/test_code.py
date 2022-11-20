import sys
import pandas as pd
from sklearn import datasets, svm, metrics, tree
from sklearn.model_selection import train_test_split
sys.path.append(".")
from utils import preprocess_digits

digits = datasets.load_digits()
data, label = preprocess_digits(digits)
del digits
train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1

def train_dev_test_split(data, label, train_frac, dev_frac, random=False):

    dev_test_frac = 1 - train_frac

    if not random:
        x_train, x_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, shuffle=True,
        )
        x_test, x_dev, y_test, y_dev = train_test_split(
            x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac, shuffle=True
        )
    else:
        x_train, x_dev_test, y_train, y_dev_test = train_test_split(
            data, label, test_size=dev_test_frac, shuffle=True,random_state=42
        )
        x_test, x_dev, y_test, y_dev = train_test_split(
            x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac, shuffle=True,random_state=42
        )

    return x_train, y_train, x_dev, y_dev, x_test, y_test

list_first = list()
list_second = list()

x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(data, label, train_frac, dev_frac)

list_first.append(x_train.size)
df3 = pd.DataFrame(x_train)

x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(data, label, train_frac, dev_frac)
list_second.append(x_train.size)
df4 = pd.DataFrame(x_train)

list_first_different = list()
list_second__different= list()

x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(data, label, train_frac, dev_frac,random=True)

list_first_different.append(x_train.size)
df1 = pd.DataFrame(x_train)

x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(data, label,train_frac, dev_frac,random=True)
list_second__different.append(x_train.size)
df2 = pd.DataFrame(x_train)

def test_same_size():
    assert df3.equals(df4)

def test_different_size():
    assert df1.equals(df2)