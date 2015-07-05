//
//  RandomForest.h
//  RandomForest
//
//  Created by wc on 15/6/11.
//  Copyright (c) 2015年 wc. All rights reserved.
//

#ifndef __RandomForest__RandomForest__
#define __RandomForest__RandomForest__

#include "DesionTree.h"
#include "Util.h"
#include <mpi.h>

// This three utils are used only if the util file can't use.
#define TREENUM 500         // The tree number of the forest
#define DATASCALE 0.8         // The data set scale for each tree
#define FEATURESCALE 0.15    // The feature scale for each tree


#define ISCOMBINATEFEATURECOL false  // If do feature combinate operation
#define COMPOSOPERATION + // The operation of feature combinate

typedef vector<DesionTree*> Forest;

class RandomForest {
public:
    RandomForest();
    ~RandomForest();
    
    void train(Mat& mat, Row &featureName, int argc, char *argv[]);
    
    vector<ElementType> predict(Mat &mat, Row &featureName,
                                int argc, char *argv[]);
    
    void deleteForest();
    
private:
    Forest trees;
    vector<int> composIndex;
    
    int trainNum;
    double dataScale;
    double featureScale;
    bool IsCombinateFeatureCol;
    
    
private:
    void getDataSet(Mat &sourceMat, Mat &aimMat,
                    Row &featureName, Row &aimFeature,
                    double DataSetScale, double featureScale,
                    int seed);
    vector<ElementType> getLabelResult(vector< Row > &result);
    ElementType getResult(map<ElementType, int> &labelNum);
    void composFeature(Mat &sourceMat, Row &featureName, vector<int> &composIndex);
    void _composFeature(Mat &sourceMat, Row &featureName, int index1, int index2);
    
    vector<int> getComposFeatureIndex(Row &featureName, int seed);
};

#endif /* defined(__RandomForest__RandomForest__) */
