import utils.createFeature as cF
from utils import dataprep as dp
import classifiers.LSTM.model_generator as mgLSTM

import classifiers.SVM.model_generator as mgSVM

dataDir = "data\\individual_all"

if __name__ == '__main__':
    # points = dp.loadPoints()
    # cF.angle(points)
    data_x, data_y = dp.loadData(dataDir)
    nFolds = dp.createFolds(data_x,data_y,5)
    for i in range(0,len(nFolds)):
        train = nFolds[i][0]
        validation = nFolds[i][1]
        model = mgLSTM.generateModel(train[0],train[1],validation[0],validation[1])
        # print "Fold: " + str(i) + " Accuracy score: " + str(acc_scr) + "\n\n"





# parameters to do a grid search on SVM
svm_tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5, 1e-6],
                     'C': [1, 10, 100, 1000, 10000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000, 10000]}]

dataDir = "data\\individual_all"

def SvmClassification() :
    # points = dp.loadPoints()
    # cF.angle(points)
    data_x, data_y = dp.loadData(dataDir)
    nFolds = dp.createFolds(data_x,data_y,5)
    for i in range(0,len(nFolds)):
        train = nFolds[i][0]
        validation = nFolds[i][1]
        model, clf_rpt, cm, acc_scr, best_params = mgSVM.generateModel(train[0],train[1],validation[0],validation[1], svm_tuned_parameters)
        print "Fold: " + str(i) + " Accuracy score: " + str(acc_scr) + "\n\n"