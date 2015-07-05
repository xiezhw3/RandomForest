#! /usr/bin/python
# coding: utf-8

import random

# Read data from file
def getdata(filepath):
	fr = open(filepath)
	data = []
	for line in fr.readlines():
		data.append(line)
	return data

# This function is to select 30% of the
#		source data from source data set
#		randomly to make test data set.
def getCVTestData(dataset):
	dataSetSize = len(dataset)
	testNum = dataSetSize * 0.3
	testIndex = set()

	testData = []
	while len(testIndex) < testNum:
		index = int(random.uniform(1, dataSetSize))
		testIndex.add(index)

	testData = []
	testData.append(dataset[0])
	for index in testIndex:
		testData.append(dataset[index])

	trainData = []
	trainData.append(dataset[0])
	for i in range(1, dataSetSize):
		if i not in testIndex:
			trainData.append(dataset[i])

	return trainData, testData

# Write data set to file.
def writeToFile(dataSet, filepath):
	fw = open(filepath, 'w')
	for line in dataSet:
		fw.write(line)
	fw.close();

if __name__ == '__main__':
	print 'Spliting the data into train data and cv test data...'
	dataSet = getdata('../data/train.csv')
	trainData, testData = getCVTestData(dataSet)
	writeToFile(trainData, '../data/trainFin.csv')
	writeToFile(testData, '../data/testCV.csv')