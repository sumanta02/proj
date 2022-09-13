#import Modules

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew

from copy import copy, deepcopy
import warnings
pd.set_option('display.max.columns', None)

from tkinter import ttk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import os
from os import listdir

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC

from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score,auc


#initializing the main window
mainWindow = Tk()
mainWindow.title("MainWindow")
mainWindow.geometry("600x600")

#initializing the logging window
resultWindow = Toplevel(mainWindow)
resultWindow.title("Log Window")
resultWindow.geometry("600x600")
logTable = Text(resultWindow, width=70, height=30)
logTable.grid(row=0, column=0, padx=10, pady=10)

def saveResults():
    global logTable
    value=(logTable.get("1.0","end-1c"))
    resultFile = open("result.txt", "w")
    resultFile.write(value)
    resultFile.close
    
saveButton = Button(resultWindow, text="Save results", command=saveResults, padx=5, pady=5).grid(row=7, column=0, padx=10, pady=10, ipadx=5, ipady=5)

#initializing the main frame
mainFrame = ttk.Frame(mainWindow, padding = "3 3 12 12")
mainFrame.grid(column=0, row=0, sticky=(N, W, E, S))
mainWindow.columnconfigure(0, weight=1)
mainWindow.rowconfigure(0, weight=1)

regressionSet = False
classifierSet = False
proc_Complete = False
lineNum = 1.0
filename=""

classifier_models = ["SV Classifier", "KNN Classifier", "Decision Tree Classifier", "Random Forest Classifier"]
regression_models = ["Decision Tree Regressor", "Random Forest Regressor", "KNN Regressor", "SVM Regressor"]
regression_model = StringVar()
regression_model.set(regression_models[0])
classifier_model = StringVar()
classifier_model.set(classifier_models[0])


def matrixDisp(matrix):
    global logTable, lineNum
    count = 0
    for i in matrix:
        for j in i:
            logTable.insert((lineNum + count * 0.1), j + "  ")
        lineNum = lineNum + 1



##################################Pre Processing Functions#############################################
def check_null_drop(data):
    global logTable, lineNum
    dim = list(data.shape)
    logTable.insert(lineNum, "Initial no. of rows : " + str(dim[0]) + "\n")
    lineNum = lineNum + 1
    null_list = list(data.isnull().sum())
    null_list_name = list(data.isnull())
    logTable.insert(lineNum, "The Columns having null values : \n")
    lineNum = lineNum + 1
    for i in range(len(null_list)):
        if null_list[i] != 0:
            temp = null_list_name[i]
            logTable.insert(lineNum, str(temp))
            lineNum = lineNum + 1
    logTable.insert(lineNum, "Dropping the rows having null values!\n")
    lineNum = lineNum + 1
    data = data.dropna()
    data = data.reset_index(drop=True)
    dim = list(data.shape)
    logTable.insert(lineNum, "No. of rows after dropping the NaN values : " + str(dim[0] + "\n"))
    lineNum = lineNum + 1
    return data


def check_null_replace(data):
    dim = list(data.shape)
    logTable.insert(lineNum, "Initial no. of rows : " + str(dim[0]) + "\n")
    lineNum = lineNum + 1
    null_list = list(data.isnull().sum())
    null_list_col = list(data.isnull())
    null_name = []
    logTable.insert(lineNum, "The Columns having null values : \n")
    lineNum = lineNum + 1
    for i in range(len(null_list)):
        if null_list[i] != 0:
            null_name.append(null_list_col[i])
    print(null_name)
    print("")
    logTable.insert(lineNum, "Replacing all NaN values with median.\n")
    lineNum = lineNum + 1
    for name in null_name:
        data[name].fillna(data[name].median(),inplace=True)
    return data


def distribution_plot(data):
    global logTable, lineNum
    logTable.insert(lineNum, "The Data Distribution is plotted and saved as .png file\n")
    lineNum = lineNum + 1
    data.hist(figsize=(25,16))
    plt.savefig('images/dist_plot.png')
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
    global logTable, lineNum
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
        logTable.insert(lineNum, "Skewness of parameters reduced by squaring each value.\n")
        lineNum = lineNum + 1
        return data_sq_rem
    elif(len(col_sqrt_temp) < len(col_sq_temp)):
        logTable.insert(lineNum, "Skewness of parameters reduced by squareroot each value.\n")
        lineNum = lineNum + 1
        return data_sqrt_rem
    else:
        logTable.insert(lineNum, "Skewness of parameters reduced by squaring each value.\n")
        lineNum = lineNum + 1
        return data_sq_rem





 #######################################################################################################################################       

#####################################Regression Functions#####################################################################

def train_test_split_function(data_without_output, data_with_output):
    real_x = data_without_output.values
    real_y = data_with_output.iloc[:,-1].values

    std_scl = StandardScaler()

    real_x = std_scl.fit_transform(real_x)

    x_train, x_test, y_train, y_test = train_test_split(real_x, real_y, test_size=0.15)

    return(x_train, x_test, y_train, y_test)


def DT_Regressor(x_train, x_test, y_train, y_test):
    global lineNum, logTable
    dtr = DecisionTreeRegressor()
    dtr.fit(x_train, y_train)
    accuracy = dtr.score(x_test,y_test)
    y_predict_dtr = dtr.predict(x_test)
    confusion_matrix_dtr = confusion_matrix(y_test,y_predict_dtr)

    logTable.insert(lineNum, "Decision Tree Regressor : " + str(accuracy) + "\n")
    lineNum = lineNum + 1 
    logTable.insert(lineNum, "\nConfusion Matrix : \n")
    lineNum = lineNum + 1 
    matrixDisp(confusion_matrix_dtr)


def RF_Regressor(x_train, x_test, y_train, y_test):
    rfr = RandomForestRegressor()
    rfr.fit(x_train, y_train)
    accuracy = rfr.score(x_test,y_test)
    y_predict_rfr = rfr.predict(x_test)
    confusion_matrix_rfr = confusion_matrix(y_test,y_predict_rfr)

    logTable.insert(lineNum, "Random Forest Regressor : " + str(accuracy) + "\n")
    lineNum = lineNum + 1 
    logTable.insert(lineNum, "\nConfusion Matrix : \n")
    lineNum = lineNum + 1 
    matrixDisp(confusion_matrix_rfr)


def KNN_Regressor(x_train, x_test, y_train, y_test):
    knn_reg = KNeighborsRegressor()
    knn_reg.fit(x_train, y_train)
    accuracy = knn_reg.score(x_test,y_test)
    y_predict_knn_reg = knn_reg.predict(x_test)
    confusion_matrix_knn_reg = confusion_matrix(y_test,y_predict_knn_reg)

    logTable.insert(lineNum, "K-Nearest Neighbour Regressor : " + str(accuracy) + "\n")
    lineNum = lineNum + 1 
    logTable.insert(lineNum, "\nConfusion Matrix : \n")
    lineNum = lineNum + 1 
    matrixDisp(confusion_matrix_knn_reg)
    


def SVM_Regressor(x_train, x_test, y_train, y_test):
    svm_reg = SVR()
    svm_reg.fit(x_train, y_train)
    accuracy = svm_reg.score(x_test,y_test)
    y_predict_svm_reg = svm_reg.predict(x_test)
    confusion_matrix_svm_reg = confusion_matrix(y_test,y_predict_svm_reg)

    logTable.insert(lineNum, "Support Vector Regressor : " + str(accuracy) + "\n")
    lineNum = lineNum + 1 
    logTable.insert(lineNum, "\nConfusion Matrix : \n")
    lineNum = lineNum + 1 
    matrixDisp(confusion_matrix_svm_reg)


#########################################################################################################################################

###############################Classification Functions##################################################################################

def train_test_split_function(data_without_output, data_with_output):
    real_x = data_without_output.values
    real_y = data_with_output.iloc[:,-1].values

    std_scl = StandardScaler()

    real_x = std_scl.fit_transform(real_x)

    x_train, x_test, y_train, y_test = train_test_split(real_x, real_y, test_size=0.15)

    return(x_train, x_test, y_train, y_test)


def sv_classifier(x_train,x_test,y_train,y_test):
    linear_model=SVC(kernel='linear',C=1).fit(x_train,y_train)
    predictor=linear_model.predict(x_test)
    acc=linear_model.score(x_test,y_test)
    confusion_matrix_svc = confusion_matrix(y_test,predictor)

    logTable.insert(lineNum, "Accuracy of the model= " + str(acc * 100) + "%\n")
    lineNum = lineNum + 1 
    logTable.insert(lineNum, "\nConfusion Matrix : \n")
    lineNum = lineNum + 1 
    matrixDisp(confusion_matrix_svc)



def knn_classifier(x_train, x_test, y_train, y_test):
    knn_model=KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)
    pred=knn_model.predict(x_test)
    acc=knn_model.score(x_test,y_test)
    confusion_matrix_knn = confusion_matrix(y_test,pred)

    logTable.insert(lineNum, "Accuracy of the model= " + str(acc * 100) + "%\n")
    lineNum = lineNum + 1 
    logTable.insert(lineNum, "\nConfusion Matrix : \n")
    lineNum = lineNum + 1 
    matrixDisp(confusion_matrix_knn)


def dtr_classifier(x_train, x_test, y_train, y_test):
    model=DecisionTreeClassifier(criterion='entropy', max_depth=3).fit(x_train,y_train)
    pred=model.predict(x_test)
    # pred_arr=[y[i] for i in pred]
    # print(pred_arr)
    acc=model.score(x_test,y_test)
    confusion_matrix_dtrc = confusion_matrix(y_test,pred)

    logTable.insert(lineNum, "Accuracy of the model= " + str(acc * 100) + "%\n")
    lineNum = lineNum + 1 
    logTable.insert(lineNum, "\nConfusion Matrix : \n")
    lineNum = lineNum + 1 
    matrixDisp(confusion_matrix_dtrc)


def rf_classifier(x_train, x_test, y_train, y_test):
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    acc = rfc.score(x_test,y_test)
    y_predict_rfc = rfc.predict(x_test)
    confusion_matrix_rfc = confusion_matrix(y_test,y_predict_rfc)

    logTable.insert(lineNum, "Accuracy of the model= " + str(acc * 100) + "%\n")
    lineNum = lineNum + 1 
    logTable.insert(lineNum, "\nConfusion Matrix : \n")
    lineNum = lineNum + 1 
    matrixDisp(confusion_matrix_rfc)

#########################################################################################################################################


def openFile():
    global fileEntry, mainFrame, regressionSet, classifierSet, filename, logTable, lineNum
    filename = filedialog.askopenfilename(initialdir=".", title="Select your dataset", filetypes=(("Dataset", "*.csv"),))
    fileEntry['state'] = NORMAL
    fileEntry.delete(0, END)
    fileEntry.insert(0, str(filename))
    logTable.insert(lineNum, "Selected file: " + str(filename) + "\n")
    lineNum = lineNum + 1
    selectionLabel = Label(mainFrame, text="Choose type of model",anchor="nw").grid(row=2, column=0)
    Radiobutton(mainFrame, text = "Regression", variable=regression_model, value="regression", command=regressionSelection).grid(row=3, column=0, sticky=W)
    Radiobutton(mainFrame, text = "Classifier", variable=classifier_model, value="classifier", command=classifierSelection).grid(row=4, column=0, sticky=W)
    return

explorerIcon = PhotoImage(file="iconFiles/file.png")
inputLabel = Label(mainFrame, text="Enter your database file", anchor="w", padx=3, pady=3)
fileEntry = Entry(mainFrame, width="30", relief=SUNKEN, state=DISABLED)
explorerButton = Button(mainFrame, image=explorerIcon, width=30, height=30, command = openFile)

inputLabel.grid(row=0, column=0)
fileEntry.grid(row=1, column=0, columnspan=3, padx=3, pady=3)
explorerButton.grid(row=1, column=3, padx=3, pady=3)


regressionLabel = Label(mainFrame, text="Choose type of regression model")
regressionDrop = OptionMenu(mainFrame, regression_model, *regression_models)
classifierLabel = Label(mainFrame,text="Choose type of classification model")
classifierDrop = OptionMenu(mainFrame, classifier_model, *classifier_models)

modelType = StringVar()
modelType.set("regression")

def classifierSelection():
    global classifier_model, classifierLabel, classifierDrop, regressionSet, regressionLabel, classifierSet, start
    if (regressionSet):
        regressionDrop.grid_remove()
        regressionLabel.grid_remove()
    classifierLabel.grid(row=5, column=0, sticky=W, pady=5)
    classifierDrop.grid(row=6, column=0, sticky=W)
    regressionSet = False
    classifierSet = True
    start = Button(mainFrame, text="Start Processing", padx=5, pady=5, command=start_proc).grid(row=8, column=0, ipadx=10, ipady=10)
    return

def regressionSelection():
    global regression_model, regressionDrop, regressionLabel, regressionSet, classifierSet, classifierLabel, start
    if (classifierSet):
        classifierDrop.grid_remove()
        classifierLabel.grid_remove()
    regressionLabel.grid(row=5, column=0, sticky=W, pady=5)
    regressionDrop.grid(row=6, column=0, sticky=W)
    regressionSet = True
    classifierSet = False
    start = Button(mainFrame, text="Start Processing", padx=5, pady=5, command=start_proc).grid(row=8, column=0, ipadx=10, ipady=10)
    return
    

def start_proc():
    global filename, regressionSet, classifierSet, regression_model, classifier_model, logTable, lineNum
    if (classifierSet):
        logTable.insert(lineNum, "Processing Started with classifier model: " + str(classifier_model.get()) + "\n")
    if (regressionSet):
        logTable.insert(lineNum, "Processing Started with regression model: " + str(regression_model.get()) + "\n")
    lineNum = lineNum + 1
    #data = pd.read_csv(filename)
    ##implement functions
    proc_Complete = True
    results = Button(mainFrame, text="Show histograms", padx=5, pady=5, command=show_results).grid(row=9, column=0, ipadx=10, ipady=10)
    resultWindow.mainloop()
    return
    
def show_results():
    imgWindow = Toplevel(mainWindow)
    imgWindow.title("Image Viewer")
    imgWindow.geometry("600x600")
    curr = ImageTk.PhotoImage(Image.open("images/dist_plot.png"))
    currImage = Label(imgWindow, image=curr)
    currImage.image = curr
    currImage.grid(row=0, column=0, columnspan=3)
    imgWindow.mainloop()
    return



mainWindow.mainloop()