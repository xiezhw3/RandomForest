//
//  Test.h
//  RandomForest
//
//  Created by wc on 15/6/11.
//  Copyright (c) 2015年 wc. All rights reserved.
//

#ifndef __RandomForest__Test__
#define __RandomForest__Test__

#include <iostream>
#include "FileProcesser.h"
#include "DesionTree.h"
#include "RandomForest.h"
#include "tools.h"
#include <cmath>
#include "Util.h"

#define CVTESTTRAINFILE "../data/trainFin.csv"
#define CVTESTTESTFILE "../data/testCV.csv"
#define TRAINFILE "../data/train.csv"
#define TESTFILE "../data/test.csv"
#define RESULTFILE "../result/result.csv"

#define DELIM ','

#define FEATURETHRESHOD 0.001

void initMpiAndGetInfo(int argc, char *argv[], int &rank, int &size);
void test(int argc, char *argv[]);
void runMain(int argc, char *argv[]);
void choseFeature(int argc, char *argv[]);

void readDate(int argc, char *argv[], const string &filepath,
              vector< vector<string> > &mat);

void runMainWithDeleteFeature(int argc, char *argv[],
                              vector<int> &featureDeleteIndexList);

void testWithDeleteFeature(int argc, char *argv[],
                           vector<int> &featureDeleteIndexList);


vector<int> getFeatureDeleteIndex(multimap<double, int> &errorRateIndex,
                                  double threshod);

#endif /* defined(__RandomForest__Test__) */
