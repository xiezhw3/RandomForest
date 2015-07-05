//
//  tools.h
//  RandomForest
//
//  Created by wc on 15/6/11.
//  Copyright (c) 2015年 wc. All rights reserved.
//

#ifndef __RandomForest__tools__
#define __RandomForest__tools__

#include "FileProcesser.h"
#include "RandomForest.h"
#include <set>
#include <cstdlib>

void getMatAndFeature(vector< vector<string> > &mat, bool hasLabel,
                      Mat &dataMat, Row &featureName);

vector<int> getCVData(vector< vector<string> > &mat, Mat &dataMat, Row &featureName);

int getIntValue(ElementType value);

void disorganizeFeature(Mat &mat, int featureIndex);

void deleteFeature(Mat &dataSet, Row &featureName, vector<int> &indexList);

#endif /* defined(__RandomForest__tools__) */
