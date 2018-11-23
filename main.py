import csv

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


from imblearn.over_sampling import SMOTE

# for dataset random shuffling
from sklearn.utils import shuffle

# import sklearn metrics
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# import keras deep learning model framework
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras import optimizers

import seaborn as sns

class dataProcessor:
    def __init__(self,modelType):

        self.modelType = modelType



    def triggerTraining(self):
        if (self.modelType == 'random_forest'):
            self.trainForestModel()
        elif (self.modelType == 'neural_network'):
            self.trainKerasModel();
        elif (self.modelType == 'logistic_regression'):
            self.trainModel()
        else:
            print('bad model type, exiting')
            exit(0)


    def loadData(self):
        # nacti data
        #file = open('hmeq.csv', 'r', encoding='utf-8')
        #self.data = csv.reader(file, delimiter=',', quotechar='|')

        self.data = pd.read_csv("hmeq.csv")


    def describeData(self):

        print('peak on the data: ')
        print(self.data.head())
        # count cols

        print('counts in the dataset columns (non null values): ')
        print(self.data.count())

        # print shape of the data
        print("data shape: ")
        print(self.data.shape)

        # print categorical
        print('What are the jobs of the customers?')
        print(self.data['JOB'].unique())

        print('What are the reasons of the customers?')
        print(self.data['REASON'].unique())

        print('value counts for target class')
        print(self.data['BAD'].value_counts())

        # get mean by target class
        print('get mean value for all numerical columns by class')
        print(self.data.groupby('BAD').mean())

    def visualizeData(self):
        # visualize original data
        pd.crosstab(self.data.JOB, self.data.BAD).plot(kind='bar')
        plt.title('Deliquency based on Job')
        plt.xlabel('Job')
        plt.ylabel('Number of customers')

        plt.show()
        plt.close()

        plt.clf()
        table = pd.crosstab(self.data.REASON, self.data.BAD)
        table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
        plt.title('Deliquency based on loan reason')
        plt.xlabel('Loan reason')
        plt.ylabel('Proportion of Customers')

        plt.show()
        plt.close()


    # derive new columns
    def deriveCols(self):

        # prints loan paid vs non paid ratio on the original data
        self.getPaidVsNonPaidRatio(self.data, 'original data')

        self.data['LTV'] = self.data['LOAN'] / self.data['VALUE'] * 100

        # LTV ratio values distribution
        print("maximum value of LTV ratio is", max(self.data['LTV']))
        print("minimum value of LTV ratio is", min(self.data['LTV']))
        print("mean LTV ratio is", np.mean(self.data['LTV']))

        plt.clf()
        plt.hist(self.data['LTV'])
        plt.title('LTV histogram')
        plt.show('LTV histogram')



        # vyfiltrovani hodnot s extremnim LTV

        self.data_filtered = self.data[(self.data['LTV'] <= 90) & (self.data['LTV'] >= 5)]

        self.getPaidVsNonPaidRatio(self.data_filtered, 'ltv_filtered')

        self.data = self.data_filtered


    # dataset description
    def getPaidVsNonPaidRatio(self,data, desc):

        print('Showing paid vs non paid ration for dataset: '+desc)
        # count percentage of paid vs non paid loans
        paid_loan = len(data[data['BAD'] == 0])
        non_paid_loan = len(data[data['BAD'] == 1])
        pct_of_paid_loans = paid_loan / (paid_loan + non_paid_loan)
        print("percentage of paid loans is", pct_of_paid_loans * 100)
        pct_of_non_paid_loans = non_paid_loan / (paid_loan + non_paid_loan)
        print("percentage of not paid loans is", pct_of_non_paid_loans * 100)


    def imputeMissingValues(self):

        print('number of samples',len(self.data))

        print('percentage of missing values per column')
        print(self.data.isnull().sum()/len(self.data)*100)



        self.data["MORTDUE"].fillna(self.data.groupby('BAD')["MORTDUE"].transform('mean'), inplace=True)
        self.data["VALUE"].fillna(self.data.groupby('BAD')["VALUE"].transform('mean'), inplace=True)
        self.data["YOJ"].fillna(self.data.groupby('BAD')["YOJ"].transform('mean'), inplace=True)
        self.data["DEROG"].fillna(self.data.groupby('BAD')["DEROG"].transform('median'), inplace=True)
        self.data["DEBTINC"].fillna(self.data.groupby('BAD')["DEBTINC"].transform('mean'), inplace=True)
        self.data["CLNO"].fillna(self.data.groupby('BAD')["CLNO"].transform('mean'), inplace=True)
        self.data["NINQ"].fillna(self.data.groupby('BAD')["NINQ"].transform('mean'), inplace=True)
        self.data["CLAGE"].fillna(self.data.groupby('BAD')["CLAGE"].transform('mean'), inplace=True)
        self.data["DELINQ"].fillna(self.data.groupby('BAD')["DELINQ"].transform('mean'), inplace=True)

        # impute categorical variables
        job = self.data[(self.data['BAD'] == 1)]['JOB'].value_counts().index[0]

        reason = self.data[(self.data['BAD'] == 1)]['REASON'].value_counts().index[0]


        self.data['JOB'].fillna(job,inplace=True)
        self.data['REASON'].fillna(reason, inplace=True)

        print('percentage of missing data in columns after imputing')
        print(self.data.isnull().sum() / len(self.data) * 100)






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

    def performSMOTE(self):

        self.data = self.data.dropna()

        print('original data',len(self.data))

        X = self.data.loc[:, self.data.columns != 'BAD']
        Y = self.data.loc[:, self.data.columns == 'BAD']



        os = SMOTE(random_state=0)

        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

        columns = X_train.columns

        os_data_X, os_data_y = os.fit_sample(X_train, y_train)

        os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
        os_data_y = pd.DataFrame(data=os_data_y, columns=['y'])
        # we can Check the numbers of our data
        print("length of oversampled data is ", len(os_data_X))
        print("Number of no subscription in oversampled data", len(os_data_y[os_data_y['y'] == 0]))

        print("Number of subscription", len(os_data_y[os_data_y['y'] == 1]))
        print("Proportion of no subscription data in oversampled data is ",
              len(os_data_y[os_data_y['y'] == 0]) / len(os_data_X))
        print("Proportion of subscription data in oversampled data is ",
              len(os_data_y[os_data_y['y'] == 1]) / len(os_data_X))

        # passing oversampled data for training
        self.XData = os_data_X
        self.YData = os_data_y




    def trainModel(self):

        print('all samples',len(self.data))
        self.data = self.data.dropna()

        print('non null samples')

        '''
        X = self.data.loc[:, self.data.columns != 'BAD']
        Y = self.data.loc[:, self.data.columns == 'BAD']
        '''

        X = self.XData
        Y = self.YData

        X_train, X_test, y_train, y_test = train_test_split(X, Y.values.ravel(), test_size=0.3, random_state=0)


        logreg = LogisticRegression()

        logreg.fit(X_train, y_train)

        y_pred = logreg.predict(X_test)
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

        self.evaluateModel(y_test,y_pred,X_test,logreg)

    def trainKerasModel(self):

        self.data = self.data.dropna()

        max_size = self.data['BAD'].value_counts().max()

        print(max_size)

        '''
        lst = [self.data]
        for class_index, group in self.data.groupby('BAD'):
            lst.append(group.sample(max_size - len(group), replace=True))
        new = pd.concat(lst)
        '''


        self.data = shuffle(self.data)
        X = self.data.loc[:, self.data.columns != 'BAD']
        Y = self.data.loc[:, self.data.columns == 'BAD']


        #X = self.XData
        #Y = self.YData

        X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values.ravel(), test_size=0.2, random_state=0)

        print('sum of non_paid_loans in y_train and y_test: ',np.sum(y_train),np.sum(y_test))

        model = Sequential()
        #model.add(Dropout(0.2, input_shape=(18,)))
        model.add(Dense(19, input_dim=19, kernel_initializer='normal', activation='relu'))
        #model.add(Dropout(0.2))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(56, activation='relu'))
        model.add(Dense(1, activation='softmax'))
        # Compile model
        optimizers = keras.optimizers.Nadam(lr=0.001)
        model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

        checkpoint = ModelCheckpoint('modelOneHot.h5', monitor='accuracy', save_best_only=True, mode='min')


        model.fit(X_train, y_train, epochs=5, batch_size=128,callbacks=[checkpoint])

        y_pred = model.predict(X_test).ravel().astype(int)


        print(y_pred)

        print('predicted sum',np.sum(y_pred))

        from sklearn.metrics import confusion_matrix
        confusion_matrix = confusion_matrix(y_test, y_pred)
        print(confusion_matrix)

        #self.evaluateModel(y_test, y_pred, X_test, model)



        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)


        auc_keras = auc(fpr_keras, tpr_keras)

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()

    def trainForestModel(self):
        self.data = self.data.dropna()

        X = self.data.loc[:, self.data.columns != 'BAD']
        Y = self.data.loc[:, self.data.columns == 'BAD']

        X_train, X_test, y_train, y_test = train_test_split(X, Y.values.ravel(), test_size=0.2, random_state=0)

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


        print(classification_report(y_test, y_pred))


        logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
        fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, label=self.modelType+' (area = %0.2f)' % logit_roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('Log_ROC')
        plt.show()


def triggerDataProcessor():

    # logistic_regression, random_forest, neural_network
    modelType = 'random_forest'

    dataTransformer = dataProcessor(modelType);

    # load data into
    dataTransformer.loadData()

    # describe data
    dataTransformer.describeData()

    # visualize data
    dataTransformer.visualizeData()

    # impute missing values
    # imputing missing values increases accuracy
    dataTransformer.imputeMissingValues()

    # creates new rows based on existing ones
    # creates LTV column and loan percentage paid column
    dataTransformer.deriveCols()

    # encode categorical variables into dummy variables
    dataTransformer.createDummy()

    # SMOTE upsampling for class balancing
    dataTransformer.performSMOTE()

    # trigger training of the model, based on the selected model types
    # evaluation of the model is also performed in this method
    dataTransformer.triggerTraining()


triggerDataProcessor()


