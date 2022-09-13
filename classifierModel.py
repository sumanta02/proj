from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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

    print("Accuracy of the model=",acc*100,"%")
    print("The confusion matrix: \n",confusion_matrix_svc)



def knn_classifier(x_train, x_test, y_train, y_test):
    knn_model=KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)
    pred=knn_model.predict(x_test)
    acc=knn_model.score(x_test,y_test)
    confusion_matrix_knn = confusion_matrix(y_test,pred)

    print("Accuracy of the model=",acc*100,"%")
    print("The confusion matrix: \n",confusion_matrix_knn)


def dtr_classifier(x_train, x_test, y_train, y_test):
    model=DecisionTreeClassifier(criterion='entropy', max_depth=3).fit(x_train,y_train)
    pred=model.predict(x_test)
    # pred_arr=[y[i] for i in pred]
    # print(pred_arr)
    acc=model.score(x_test,y_test)
    confusion_matrix_dtrc = confusion_matrix(y_test,pred)
    print("Decision tree model accuracy=",acc*100)
    print("The Confusion matrix: \n",confusion_matrix_dtrc)


def rf_classifier(x_train, x_test, y_train, y_test):
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    accuracy = rfc.score(x_test,y_test)
    y_predict_rfc = rfc.predict(x_test)
    confusion_matrix_rfc = confusion_matrix(y_test,y_predict_rfc)

    print("Random Forest Classifier : ",accuracy)
    print("\nConfusion Matrix : \n",confusion_matrix_rfc)