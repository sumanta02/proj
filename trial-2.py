#import Modules

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew

from copy import copy, deepcopy
import warnings

from tkinter import ttk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import os
from os import listdir

#initializing the main window
mainWindow = Tk()
mainWindow.title("MainWindow")
mainWindow.geometry("600x600")

#initializing the main frame
mainFrame = ttk.Frame(mainWindow, padding = "3 3 12 12")
mainFrame.grid(column=0, row=0, sticky=(N, W, E, S))
mainWindow.columnconfigure(0, weight=1)
mainWindow.rowconfigure(0, weight=1)

def openFile():
    global fileEntry, mainFrame
    mainWindow.filename = filedialog.askopenfilename(initialdir=".", title="Select your dataset", filetypes=(("Dataset", "*.csv"),))
    fileEntry['state'] = NORMAL
    fileEntry.delete(0, END)
    fileEntry.insert(0, str(mainWindow.filename))
    selectionLabel = Label(mainFrame, text="Choose type of model",anchor="nw").grid(row=2, column=0)
    Radiobutton(mainFrame, text = "Regression", variable=modelType, value="regression", command=regressionSelection).grid(row=3, column=0, sticky=W)
    Radiobutton(mainFrame, text = "Classifier", variable=modelType, value="classifier", command=classifierSelection).grid(row=4, column=0, sticky=W)
    return

explorerIcon = PhotoImage(file="iconFiles/file.png")
inputLabel = Label(mainFrame, text="Enter your database file", anchor="w", padx=3, pady=3)
fileEntry = Entry(mainFrame, width="30", relief=SUNKEN, state=DISABLED)
explorerButton = Button(mainFrame, image=explorerIcon, width=30, height=30, command = openFile)

inputLabel.grid(row=0, column=0)
fileEntry.grid(row=1, column=0, columnspan=3, padx=3, pady=3)
explorerButton.grid(row=1, column=3, padx=3, pady=3)


classifier_models = ["SV Classifier", "KNN Classifier", "Deep Tree Classifier", "Random Forest Classifier"]
regression_models = ["Decision Tree Regressor", "Random Forest Regressor", "KNN Regressor", "SVM Regressor"]
regression_model = StringVar()
regression_model.set(regression_models[0])
classifier_model = StringVar()
classifier_model.set(classifier_models[0])
regressionSet = False
classifierSet = False
regressionLabel = Label(mainFrame, text="Choose type of regression model")
regressionDrop = OptionMenu(mainFrame, regression_model, *regression_models)
classifierLabel = Label(mainFrame,text="Choose type of classification model")
classifierDrop = OptionMenu(mainFrame, classifier_model, *classifier_models)

modelType = StringVar()
modelType.set("regression")

def classifierSelection():
    global classifier_model, classifierLabel, classifierDrop, regressionSet, regressionLabel, classifierSet
    if (regressionSet):
        regressionDrop.grid_remove()
        regressionLabel.grid_remove()
    classifierLabel.grid(row=5, column=0, sticky=W, pady=5)
    classifierDrop.grid(row=6, column=0, sticky=W)
    regressionSet = False
    classifierSet = True

def regressionSelection():
    global regression_model, regressionDrop, regressionLabel, regressionSet, classifierSet, classifierLabel
    if (classifierSet):
        classifierDrop.grid_remove()
        classifierLabel.grid_remove()
    regressionLabel.grid(row=5, column=0, sticky=W, pady=5)
    regressionDrop.grid(row=6, column=0, sticky=W)
    regressionSet = True
    classifierSet = False
    

def start_proc():
    return






mainWindow.mainloop()