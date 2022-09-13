import tkinter
from tkinter.ttk import *
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import os
from os import listdir

root = Tk()
root.title("Trial 1")
root.geometry("400x400")

fileIcon = PhotoImage(file="file.png")
e = Entry(root, width="30", borderwidth=1, relief=SUNKEN)

linearModels = ["model1", "model2", "model3"]
classificationModels = ["model1", "model2", "model3"]
linearModel = StringVar()
linearModel.set(linearModels[0][1])
classificationModel = StringVar()
classificationModel.set(linearModels[0][1])
linearSet = False
classificationSet = False
linearLabel = Label(text="Choose type of linear model")
linearDrop = OptionMenu(root, linearModel, *linearModels)
classificationLabel = Label(text="Choose type of classification model")
classificationDrop = OptionMenu(root, classificationModel, *classificationModels)
images = []
img_folder = "images"
for image in os.listdir(img_folder):
    if image.endswith(".png"):
        images.append("images/" + str(image))

def openFile():
    root.filename = filedialog.askopenfilename(initialdir=".", title="Select your dataset", filetypes=(("Dataset", "*.csv"),))
    e.delete(0, END)
    e.insert(0, str(root.filename))

label1 = Label(text="Select your dataset",anchor="w").grid(row=0, column=0)
e.grid(row=1, column=0, sticky="EW", ipady=5, padx=5)
fileButton = Button(root, image=fileIcon, width=32, height=32, command = openFile).grid(row=1, column=1)

def linearSelection():
    global linearModel, linearDrop, linearLabel, linearSet, classificationSet, classificationLabel
    if (classificationSet):
        classificationDrop.grid_remove()
        classificationLabel.grid_remove()
    linearLabel.grid(row=5, column=0, sticky=W, pady=5)
    linearDrop.grid(row=6, column=0, sticky=W)
    linearSet = True
    classificationSet = False

def classificationSelection():
    global classificationModel, classificationLabel, classificationDrop, linearSet, linearLabel, classificationSet
    if (linearSet):
        linearDrop.grid_remove()
        linearLabel.grid_remove()
    classificationLabel.grid(row=5, column=0, sticky=W, pady=5)
    classificationDrop.grid(row=6, column=0, sticky=W)
    linearSet = False
    classificationSet = True

def displayImages():
    global images
    imgWindow = Toplevel(root)
    imgWindow.title("Image Viewer")
    imgWindow.geometry("600x600")
    curr = ImageTk.PhotoImage(Image.open("images/0.png"))
    currImage = Label(imgWindow, image=curr)
    currImage.image = curr
    currImage.grid(row=0, column=0, columnspan=3)
    def forward(image_number):
        global images
        global curr
        global currImage
        global button_forward
        global button_back
        curr = ImageTk.PhotoImage(Image.open("images/1.png"))
        currImage = Label(imgWindow, image=curr)
        currImage.image = curr
        button_forward = Button(imgWindow, text=">>", command=lambda: forward(image_number+1))
        button_back = Button(imgWindow, text="<<", command=lambda: back(image_number-1))
        
        currImage.grid(row=0, column=0, columnspan=3)
        button_back.grid(row=1, column=0)
        button_forward.grid(row=1, column=2)

    def back(image_number):
        global images
        global currImage
        global button_forward
        global button_back

        curr = ImageTk.PhotoImage(Image.open("images/2.png"))
        currImage = Label(imgWindow, image=curr)
        currImage.image = curr
        button_forward = Button(imgWindow, text=">>", command=lambda: forward(image_number+1))
        button_back = Button(imgWindow, text="<<", command=lambda: back(image_number-1))
        
        currImage.grid(row=0, column=0, columnspan=3)
        button_back.grid(row=1, column=0)
        button_forward.grid(row=1, column=2)
    
    button_back = Button(imgWindow, text="<<", command=back)
    button_exit = Button(imgWindow, text="Exit Program", command=imgWindow.destroy)
    button_forward = Button(imgWindow, text=">>", command=lambda: forward(2))

    button_back.grid(row=1, column=0)
    button_exit.grid(row=1, column=1)
    button_forward.grid(row=1, column=2)
    

modelType = StringVar()
modelType.set("Linear")
label1 = Label(text="Choose type of model",anchor="ne").grid(row=2, column=0)
Radiobutton(root, text = "Linear", variable=modelType, value="linear", command=linearSelection).grid(row=3, column=0, sticky=W)
Radiobutton(root, text = "Classification", variable=modelType, value="classification", anchor="w", command=classificationSelection).grid(row=4, column=0, sticky=W)

run = Button(root, text="Run Model", padx=5, pady=5).grid(row=8, column=0, ipadx=10, ipady=10)
showImages = Button(root, text="Show Images", padx=5, pady=5, command=displayImages).grid(row=9, column=0, ipadx=10, ipady=10)


root.mainloop()