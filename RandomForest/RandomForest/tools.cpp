//
//  tools.cpp
//  RandomForest
//
//  Created by wc on 15/6/11.
//  Copyright (c) 2015年 wc. All rights reserved.
//

#include "tools.h"

// This function is to get the cv ElementType type data set and featureName.
// The parameter if from file reader.
// This function change the data type from string to ElementType
void getMatAndFeature(vector< vector<string> > &mat, bool hasLabel, Mat &dataMat, Row &featureName) {
    assert(mat.size() > 0);
    
    int rowSize = (int)mat[0].size();
    for (int i = 0; i < mat.size(); ++i) {
        if (mat[i].size() == 0)
            break;
        // The feature name
        if (i == 0) {
            int featureNum = rowSize;
            if (hasLabel)
                featureNum -= 1;
            
            for (int j = 1; j < featureNum; ++j) {
                featureName.push_back(j);
            }
            // The feature data and label.
        } else {
            Row row;
            for (int j = 1; j < rowSize; ++j) {
                // We change the data type into double.
                // In my program, I make double ElementType
                // because the feature value in data set is double type.
                row.push_back(std::stod(mat[i][j]));
            }
            
            dataMat.push_back(row);
        }
    }
}

// This function is to get the cv test data from cv test file.
// I split the train file into new train file and cv test file
//      use python code.
vector<int> getCVData(vector< vector<string> > &mat, Mat &dataMat, Row &featureName) {
    int rowSize = (int)mat[0].size();
    vector<int> labelata;
    
    for (int i = 0; i < mat.size(); ++i) {
        if (mat[i].size() == 0)
            break;
        if (i == 0) {
            for (int j = 1; j < rowSize - 1; ++j) {
                featureName.push_back(j);
            }
        } else {
            Row row;
            int j = 1;
            for (; j < rowSize - 1; ++j) {
                row.push_back(std::stod(mat[i][j]));
            }
            labelata.push_back(std::stoi(mat[i][j]));
            
            dataMat.push_back(row);
        }
    }
    
    return labelata;
}

// This function is to disorganize the secquence of
//       the feature in featureIndex randomly.
void disorganizeFeature(Mat &mat, int featureIndex) {
    int size = (int)mat.size();
    srand((unsigned)time(NULL) * (featureIndex + 1));
    
    vector<int> indexTemp;
    for (int i = 0; i < size; ++i)
        indexTemp.push_back(i);
    
    vector<int> indexMap;
    while (indexTemp.size() > 0) {
        int index = (int)(rand() % indexTemp.size());
        indexMap.push_back(indexTemp[index]);
        indexTemp.erase(indexTemp.begin() + index);
    }
    
    vector<int> indexFeature;
    for (auto index : indexMap)
        indexFeature.push_back(mat[index][featureIndex]);
    
    for (int i = 0; i < size; ++i) {
        mat[i][featureIndex] = indexFeature[i];
    }
}

// This function if to delete the feature with index in indexList.
void deleteFeature(Mat &dataSet, Row &featureName, vector<int> &indexList) {
    if (dataSet.size() == 0)
        return;
    
    Mat matTemp = dataSet;
    Row featureNameTemp = featureName;
    
    dataSet.clear();
    featureName.clear();
    
    int dataRowSize = (int)matTemp[0].size();
    vector<int> indexIsDelete(dataRowSize);
    
    for (int i = 0; i != dataRowSize; ++i)
        indexIsDelete[i] = 0;
    
    for (auto item : indexList)
        indexIsDelete[item] = 1;
    
    for (auto row : matTemp) {
        Row rowTemp;
        for (int i = 0; i < dataRowSize; ++i) {
            if(indexIsDelete[i] == 0) {
                rowTemp.push_back(row[i]);
            }
        }
        dataSet.push_back(rowTemp);
    }
    
    for (int i = 0; i < featureNameTemp.size(); ++i) {
        if (indexIsDelete[i] == 0)
            featureName.push_back(featureNameTemp[i]);
    }
}

// Get the int value of a double number with
//      round-off method.
int getIntValue(ElementType value) {
    int result = (int)value;
    if (value - result > 0.5)
        result += 1;
    
    return result;
}