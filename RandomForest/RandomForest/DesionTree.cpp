//
//  DesionTree.cpp
//  RandomForest
//
//  Created by wc on 15/6/10.
//  Copyright (c) 2015年 wc. All rights reserved.
//

#include "DesionTree.h"

DesionTree::DesionTree() {
    this->_tree = nullptr;
}

DesionTree::~DesionTree() {
    this->deleteTree();
}

// Delete all nodes in a tree.
void DesionTree::deleteTree() {
    stack<treeNode> s;
    s.push(_tree);
    
    while (!s.empty()) {
        treeNode current = s.top();
        s.pop();
        
        if (current != nullptr) {
            if (current->_right != nullptr)
                s.push(current->_right);
            if (current->_left != nullptr)
                s.push(current->_left);
            
            delete current;
        }
    }
    _tree = nullptr;
}

/********************************************************************
 * Description: Create a CART tree with the data set.
 *
 *@para mat Data set
 *@para featureName The feature name
 ********************************************************************/
void DesionTree::builtTree(Mat &mat, Row &featureName) {
    this->_tree = new Node(mat);
    
    // Use Not Recursion method to build tree
    stack<treeNode> treeContainer;
    treeContainer.push(_tree);
    
    while (!treeContainer.empty()) {
        treeNode current = treeContainer.top();
        treeContainer.pop();
        Mat currentMat = current->_mat;
        
        if (idLabelTheSame(currentMat)) {
            // If the label of the mat is same, means reach the end of the tree.
            current->_label = getLabel(currentMat);
        } else {
            ElementType splitFeatureValue;
            int bestSplitFeatureIndex;
            
            // Use gini gain to get the best split feature and its split value.
            getBestSplitFeatureInfo(currentMat, featureName, bestSplitFeatureIndex, splitFeatureValue);
            Mat mat1, mat2;
            
            // Split the data mat to two sub mat with the split feature and its value.
            splitMat(currentMat, mat1, mat2, bestSplitFeatureIndex, splitFeatureValue);
            if (mat1.size() == 0 || mat2.size() == 0) {
                // The value of feature is the same, can not split anymore.
                current->_label = getLabel(currentMat);
            } else {
                // Create sub node.
                treeNode left = new Node(mat1, current->_level + 1);
                treeNode right = new Node(mat2, current->_level + 1);
                current->_left = left;
                current->_right = right;
                current->_splitFeatureName = featureName[bestSplitFeatureIndex];
                current->_splitFeatureValue = splitFeatureValue;
                
                treeContainer.push(right);
                treeContainer.push(left);
            }
        }
    }
    clearTree();
}

// Clear the met and featureName in each node.
void DesionTree::clearTree() {
    stack<treeNode> s;
    s.push(_tree);
    while (!s.empty()) {
        treeNode current = s.top();
        s.pop();
        
        if (current != nullptr) {
            if (current->_right != nullptr)
                s.push(current->_right);
            if (current->_left != nullptr)
                s.push(current->_left);
            
            // Clear the mat and featureName
            current->_mat.clear();
        }
    }
}


// Split the mat into two sub mat.
// Because the feature is continuous, we don't erase the feature col
//      when split.
void DesionTree::splitMat(Mat &mat,
                          Mat &mat1,
                          Mat &mat2,
                          int bestSplitFeatureIndex,
                          ElementType &splitFeatureValue) {
    mat1.clear();
    mat2.clear();
    for (auto row : mat) {
        if (row[bestSplitFeatureIndex] < splitFeatureValue) {
            mat1.push_back(row);
        } else {
            mat2.push_back(row);
        }
    }
}

// To get the best split feature and its split value of the mat.
void DesionTree::getBestSplitFeatureInfo(Mat &mat, Row &feature,
                                         int &bestSplitFeatureIndex,
                                         ElementType &splitFeatureValue) {
    assert(mat.size() > 0);
    
    int featureNum = (int)feature.size();
    double bestGiniGain = 100.0;
    bestSplitFeatureIndex = 0;
    for (int i = 0; i < featureNum; ++i) {
        ElementType splitValue;
        
        // Get the smallest gini gain of each feature.
        double featureMinGiniGain = getMinGiniGain(mat, i, splitValue);
        
        // Chose the feature with min gini gain.
        if (featureMinGiniGain < bestGiniGain) {
            bestGiniGain = featureMinGiniGain;
            splitFeatureValue = splitValue;
            bestSplitFeatureIndex = i;
        }
    }
}


// Get the min gini gain of a feature.
double DesionTree::getMinGiniGain(Mat &mat, int featureIndex, ElementType &splitValue) {
    assert(mat.size() > 0);
    
    multimap<ElementType, ElementType> featureLabel;
    map<ElementType, int> leftLabel, rightLabel;
    
    int labelIndex = (int)mat[0].size() - 1;
    
    for (auto row : mat) {
        // featureLabel is use to get a sorted feature-label map,
        // which is use to reduce the caculate.
        featureLabel.insert(pair<ElementType, ElementType>(row[featureIndex], row[labelIndex]));
        rightLabel[row[labelIndex]] += 1;
    }
    
    auto iter = featureLabel.begin();
    ElementType preFeatureValue = iter->first;
    
    rightLabel[iter->second] -= 1;
    leftLabel[iter->second] += 1;
    ++iter;
    
    ElementType middle = preFeatureValue;
    double minGiniGain = 100.0;
    splitValue = 0.0;
    
    while (iter != featureLabel.end()) {
        // Only caculate the gini gain when the feature value is change.
        if (iter->first != preFeatureValue) {
            preFeatureValue = iter->first;
            middle += iter->first;
            middle /= 2.0;
            
            // Get the gini gain in each split point.
            double currentGiniGain = getGiniGain(leftLabel, rightLabel);
            if (minGiniGain > currentGiniGain) {
                minGiniGain = currentGiniGain;
                splitValue = middle;
            }
        }
        
        rightLabel[iter->second] -= 1;
        leftLabel[iter->second] += 1;
        middle = iter->first;
        
        if (rightLabel[iter->second] == 0) {
            rightLabel.erase(rightLabel.find(iter->second));
        }
        
        ++iter;
    }
    
    return minGiniGain;
}

// Caculate the gini gain.
// In CART, we only split a node into two sub node,
// so in here we only have two sub node.
double DesionTree::getGiniGain(map<ElementType, int> &leftLabel,
                               map<ElementType, int> &rightLabel) {
    double total = 0, leftNum, rightNum;
    
    // Get the gini value of left and right node.
    double leftGini = getGini(leftLabel, leftNum);
    double rightGini = getGini(rightLabel, rightNum);
    total = leftNum + rightNum;
    
    assert(total != 0);
    return leftGini * (leftNum / total) + rightGini * (rightNum / total);
}

// Get the gini value of a data set.
double DesionTree::getGini(map<ElementType, int> &labelNum, double &total) {
    double gini = 0.0;
    total = 0.0;
    for (auto item : labelNum) {
        total += (double)item.second;
    }
    
    // The caculate formula is gini = 1 - p * p
    for (auto item : labelNum) {
        double p = (double)(item.second) / total;
        gini += p * p;
    }
    
    return 1 - gini;
}

// This function is to get the label of a mat, which is
// the label with max times in labels.
ElementType DesionTree::getLabel(Mat &mat) {
    map<ElementType, int> labelNum = getLabelNum(mat);
    
    int maxLabelNum = 0;
    ElementType maxLabel = 0.0;
    
    for (auto item : labelNum) {
        if (maxLabelNum < item.second) {
            maxLabelNum = item.second;
            maxLabel = item.first;
        }
    }
    
    return maxLabel;
}

// This function is to judge whether the labl in a data set
// is the same.
bool DesionTree::idLabelTheSame(Mat &mat) {
    map<ElementType, int> labelNum = getLabelNum(mat);
    
    return (labelNum.size() == 1);
}

// This function is to get each label's times  of labels.
// I use a map struct to store the result.
map<ElementType, int> DesionTree::getLabelNum(Mat &mat) {
    assert(mat.size() > 0);
    
    map<ElementType, int> labelNum;
    int labelIndex = (int)mat[0].size() - 1;
    
    for (auto row : mat) {
        labelNum[row[labelIndex]] += 1;
    }
    
    return labelNum;
}

// This function is to predict the result of test data set.
// It caculate the label of each row with the CART tree.
Row DesionTree::predict(Mat &mat, Row &featureName) {
    map<ElementType, int> featureIndex;
    Row result;
    
    // Map the feature name with its index.
    // And then we can get the feature name with its index.
    for (int i = 0; i < featureName.size(); ++i) {
        featureIndex[featureName[i]] = i;
    }
    
    for (auto row : mat) {
        result.push_back(classify(row, featureIndex));
    }
    
    return result;
}

// This function is to get the result of a data row.
ElementType DesionTree::classify(Row &row, map<ElementType, int> &featureIndex) {
    treeNode current = _tree;
    
    // Use dp method to get the result.
    while(current != nullptr) {
        // This means node current is the leaf of the tree.
        // And the label of this node is the predict result.
        if (current->_left == nullptr || current->_right == nullptr) {
            return current->_label;
        } else {
            int index = featureIndex[current->_splitFeatureName];
            // row[index] < current->_splitFeatureValue means we
            // should do to left sub node of node current.
            if (row[index] < current->_splitFeatureValue) {
                current = current->_left;
            } else {
                current = current->_right;
            }
        }
    }
    
    // If program reach here, means can't find a result, this means the tree we
    // created is wrong.
    ErrorMesg("\t\tError appear! The node out of tree when classify", false);
    
    return 0.0;
}
