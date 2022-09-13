import pandas as pd
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max.columns', None)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score,auc


def train_test_split_function(data_without_output, data_with_output):
    real_x = data3_without_output.values
    real_y = data_with_output.iloc[:,-1].values

    std_scl = StandardScaler()

    real_x = std_scl.fit_transform(real_x)

    x_train, x_test, y_train, y_test = train_test_split(real_x, real_y, test_size=0.15)

    return(x_train, x_test, y_train, y_test)


def DT_Regressor(x_train, x_test, y_train, y_test):
    dtr = DecisionTreeRegressor()
    dtr.fit(x_train, y_train)
    accuracy = dtr.score(x_test,y_test)
    y_predict_dtr = dtr.predict(x_test)
    confusion_matrix_dtr = confusion_matrix(y_test,y_predict_dtr)

    print("Decision Tree Regressor : ",accuracy)
    print("\nConfusion Matrix : \n",confusion_matrix_dtr)


def RF_Regressor(x_train, x_test, y_train, y_test):
    rfr = RandomForestRegressor()
    rfr.fit(x_train, y_train)
    accuracy = rfr.score(x_test,y_test)
    y_predict_rfr = rfr.predict(x_test)
    confusion_matrix_rfr = confusion_matrix(y_test,y_predict_rfr)

    print("Random Forest Regressor : ",accuracy)
    print("\nConfusion Matrix : \n",confusion_matrix_rfr)


def KNN_Regressor(x_train, x_test, y_train, y_test):
    knn_reg = KNeighborsRegressor()
    knn_reg.fit(x_train, y_train)
    accuracy = knn_reg.score(x_test,y_test)
    y_predict_knn_reg = knn_reg.predict(x_test)
    confusion_matrix_knn_reg = confusion_matrix(y_test,y_predict_knn_reg)

    print("K-Nearest Neighbour Regressor : ",accuracy)
    print("\nConfusion Matrix : \n",confusion_matrix_knn_reg)


def SVM_Regressor(x_train, x_test, y_train, y_test):
    svm_reg = SVR()
    svm_reg.fit(x_train, y_train)
    accuracy = svm_reg.score(x_test,y_test)
    y_predict_svm_reg = svm_reg.predict(x_test)
    confusion_matrix_svm_reg = confusion_matrix(y_test,y_predict_svm_reg)

    print("Support Vector Regressor : ",accuracy)
    print("\nConfusion Matrix : \n",confusion_matrix_svm_reg)