//
//  DesionTree.h
//  RandomForest
//
//  Created by wc on 15/6/10.
//  Copyright (c) 2015年 wc. All rights reserved.
//

#ifndef __RandomForest__DesionTree__
#define __RandomForest__DesionTree__

#include <vector>
#include <string>
#include <stack>
#include <map>
#include <assert.h>
#include "Message.h"

using namespace std;

typedef double ElementType;
typedef vector<ElementType> Row;
typedef vector<Row> Mat;

struct Node{
    Mat _mat;
    int _level;
    ElementType _splitFeatureName;
    ElementType _splitFeatureValue;
    ElementType _label;
    
    Node *_left;
    Node *_right;
    
    Node(Mat mat,
         int level = 1,
         ElementType splitFeatureName = 0.0,
         ElementType splitFeatureValue = 0.0,
         ElementType label = 0.0,
         Node *left = nullptr,
         Node *right = nullptr) :
        _mat(mat),
        _level(level),
        _splitFeatureName(splitFeatureName),
        _splitFeatureValue(splitFeatureValue),
        _label(label),
        _left(left),
        _right(right){}
};

typedef Node* treeNode;

class DesionTree {
public:
    DesionTree();
    ~DesionTree();
    
    void builtTree(Mat &mat, Row &featureName);
    Row predict(Mat &mat, Row &featureName);
    void deleteTree();
private:
    bool idLabelTheSame(Mat &mat);
    map<ElementType, int> getLabelNum(Mat &mat);
    ElementType getLabel(Mat &mat);
    void getBestSplitFeatureInfo(Mat &mat, Row &feature,
                                 int &bestSplitFeatureIndex,
                                 ElementType &splitFeatureValue);
    double getMinGiniGain(Mat &mat, int featureIndex, ElementType &splitValue);
    double getGini(map<ElementType, int> &labelNum, double &total);
    double getGiniGain(map<ElementType, int> &leftLabel,
                       map<ElementType, int> &rightLabel);
    void splitMat(Mat &mat,
                  Mat &mat1,
                  Mat &mat2,
                  int bestSplitFeatureIndex,
                  ElementType &splitFeatureValue);
    ElementType classify(Row &row, map<ElementType, int> &featureIndex);
    
    void clearTree();
private:
    treeNode _tree;
};

#endif /* defined(__RandomForest__DesionTree__) */
