#! /usr/bin/python
#coding: utf-8

import random
import matplotlib.pyplot as plt
from numpy import *

# Chose 10 feature index randomly.
def getColIndex():
	indexNum = 0
	indexList = []
	while indexNum < 10:
		index = int(random.uniform(1, 617))
		if index not in indexList:
			indexList.append(index)
			indexNum += 1

	return indexList

# Get the 10 feature columns chosen.
# And then combine each of them with
# 		label to make two-dimension coordinate
def getData(filePath):
	fr = open(filePath)
	colDataList = [[], [], [], [], [], [], [], [], [], []]
	lables = []
	indexList = getColIndex()

	firstLine = True
	lines = fr.readlines()
	for line in lines:
		ifGetData = random.uniform(0, 10)
		
		if firstLine:
			firstLine = False
		else:
			data = line.strip().split(',')
			lables.append(data[-1])
			for i in range(10):
				colDataList[i].append(data[indexList[i]])

	return colDataList, lables

# Draw the two-dimension coordinate into image.
def plotRes(filePath):
	colDataList, lables = getData(filePath)
	for i in range(len(colDataList)):
		scale = 70
		plt.scatter(colDataList[i], lables,  c = "red", s = scale, label = "red", alpha = 0.3, edgecolors = 'none')
		plt.ylabel('label')
		plt.xlabel('feature' + str(i))
		plt.title('featureMap' + str(i))
		plt.savefig('../version/version' + str(i) + '.png', dpi = 100)
		plt.cla()

if __name__ == '__main__':
	print 'Making version...'
	plotRes('../data/train.csv')

