import csv
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
class dataProcessor:
    def __init__(self):
        self.loadData()
        self.describeData()
        self.createDummy()

        modelType = 'forest'

        if(modelType == 'forest'):
            self.trainForestModel()
        else:
            self.trainModel()
    def loadData(self):
        # nacti data
        #file = open('hmeq.csv', 'r', encoding='utf-8')
        #self.data = csv.reader(file, delimiter=',', quotechar='|')

        self.data = pd.read_csv("hmeq.csv")


    def describeData(self):
        print(self.data.head())
        # count cols
        print(self.data.count())

        # print shape
        print("data shape")
        print(self.data.shape)

        # target vals
        print(self.data['BAD'].unique())

        print(self.data['BAD'].value_counts())


        # count percentage of paid vs non paid loans
        count_no_sub = len(self.data[self.data['BAD'] == 0])
        count_sub = len(self.data[self.data['BAD'] == 1])
        pct_of_no_sub = count_no_sub / (count_no_sub + count_sub)
        print("percentage of paid loans is", pct_of_no_sub * 100)
        pct_of_sub = count_sub / (count_no_sub + count_sub)
        print("percentage of not paid loans is", pct_of_sub * 100)

        # get mean by target class
        print(self.data.groupby('BAD').mean())
    def cleanData(self):
        # vycisti data
        for line in self.data:
            pass
            #print(line)
    def imputeMissingValues(self):
        pass
    def featurizeDataset(self):
        pass

    def createDummy(self):
        dtypes = self.data.select_dtypes(include='object').columns
        print(dtypes)
        data = self.data

        cat_vars = ['JOB', 'REASON']
        for var in cat_vars:
            cat_list = 'var' + '_' + var
            cat_list = pd.get_dummies(data[var], prefix=var)
            data1 = data.join(cat_list)
            data = data1

        cat_vars = ['JOB','REASON']
        data_vars = data.columns.values.tolist()
        to_keep = [i for i in data_vars if i not in cat_vars]

        data_final = data[to_keep]

        print(data_final.columns)

        self.data = data_final


    def trainModel(self):
        self.data = self.data.dropna()

        X = self.data.loc[:, self.data.columns != 'BAD']
        Y = self.data.loc[:, self.data.columns == 'BAD']

        X_train, X_test, y_train, y_test = train_test_split(X, Y.values.ravel(), test_size=0.3, random_state=0)


        logreg = LogisticRegression()

        logreg.fit(X_train, y_train)

        y_pred = logreg.predict(X_test)
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

        self.evaluateModel(y_test,y_pred,X_test,logreg)

    def trainForestModel(self):
        self.data = self.data.dropna()

        X = self.data.loc[:, self.data.columns != 'BAD']
        Y = self.data.loc[:, self.data.columns == 'BAD']

        X_train, X_test, y_train, y_test = train_test_split(X, Y.values.ravel(), test_size=0.3, random_state=0)

        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=100)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        self.evaluateModel(y_test, y_pred, X_test, clf)

    def evaluateModel(self,y_test,y_pred,X_test,logreg):


        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

        from sklearn.metrics import confusion_matrix
        confusion_matrix = confusion_matrix(y_test, y_pred)
        print(confusion_matrix)

        from sklearn.metrics import classification_report
        print(classification_report(y_test, y_pred))

        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import roc_curve
        logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
        fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('Log_ROC')
        plt.show()







dataTransformer = dataProcessor();
