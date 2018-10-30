import csv
import pandas as pd


class dataProcessor:
    def __init__(self):
        self.loadData()
        self.describeData()
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
        self.training_data = self.data.dro




dataTransformer = dataProcessor();
