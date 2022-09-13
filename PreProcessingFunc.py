import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew

from copy import copy, deepcopy

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max.columns', None)



def check_null_drop(data):
    dim = list(data.shape)
    print("Initial no. of rows : ",dim[0])
    null_list = list(data.isnull().sum())
    null_list_name = list(data.isnull())
    print("The Columns having null values : ")
    for i in range(len(null_list)):
        if null_list[i] != 0:
            temp = null_list_name[i]
            print(temp)
    print("Dropping the rows having null values!")
    data = data.dropna()
    data = data.reset_index(drop=True)
    dim = list(data.shape)
    print("No. of rows after dropping the NaN values : ",dim[0])
    return data



def distribution_plot(data):
    print("The Data Distribution is plotted and saved as .png file")
    data.hist(figsize=(25,16))
    plt.savefig('dist_plot.png')
    plt.close()


def skewness_square_removal(data,skewed_col_name):
    for name in skewed_col_name:
        data[name] = np.square(data[name])
    return data

def skewness_sqrt_removal(data,skewed_col_name):
    for name in skewed_col_name:
        data[name] = np.sqrt(data[name])
    return data

def skewness_check_removal(data):
    data = data.iloc[: , :-1]
    column_name = list(data.columns)
    skewed_col_name = []
    for name in column_name:
        skew_val = data[name].skew()
        if(skew_val>1.5 or skew_val<-1.5):
            skewed_col_name.append(name)

    data_sq_rem = deepcopy(data)
    data_sqrt_rem = deepcopy(data)
    data_sq_rem = skewness_square_removal(data_sq_rem,skewed_col_name)
    data_sqrt_rem = skewness_sqrt_removal(data_sqrt_rem,skewed_col_name)
    col_sq = list(data_sq_rem.columns)
    col_sqrt = list(data_sqrt_rem.columns)

    col_sq_temp,col_sqrt_temp = [],[]
    
    for name in skewed_col_name:
        t1 = data_sq_rem[name].skew()
        t2 = data_sqrt_rem[name].skew()
        if(t1>1.5 or t1<-1.5):
            col_sq_temp.append(t1)
        if(t2>1.5 or t2<-1.5):
            col_sqrt_temp.append(t2)

    if(len(col_sq_temp) < len(col_sqrt_temp)):
        print("Skewness of parameters reduced by squaring each value.")
        return data_sq_rem
    elif(len(col_sqrt_temp) < len(col_sq_temp)):
        print("Skewness of parameters reduced by squareroot each value.")
        return data_sqrt_rem
    else:
        print("Skewness of parameters reduced by squaring each value.")
        return data_sq_rem