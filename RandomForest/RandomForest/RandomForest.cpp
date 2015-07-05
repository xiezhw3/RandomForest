//
//  RandomForest.cpp
//  RandomForest
//
//  Created by wc on 15/6/11.
//  Copyright (c) 2015年 wc. All rights reserved.
//

#include "RandomForest.h"
#include <set>

// Get util message from util file.
// Initialize the util variable.
RandomForest::RandomForest() {
    string trainNumStr = Util::getInstance()->getUtil("TREENUM");
    if (trainNumStr == "") {
        trainNum = TREENUM;
    } else {
        trainNum = stoi(trainNumStr);
    }
    
    string dataScaleStr = Util::getInstance()->getUtil("DATASCALE");
    if (dataScaleStr == "") {
        dataScale = DATASCALE;
    } else {
        dataScale = stod(dataScaleStr);
    }
    
    string featureScaleStr = Util::getInstance()->getUtil("FEATURESCALE");
    if (featureScaleStr == "") {
        featureScale = FEATURESCALE;
    } else {
        featureScale = stod(featureScaleStr);
    }
    
    string IsCombinateFeatureColStr = Util::getInstance()->getUtil("ISCOMBINATEFEATURECOL");
    if (IsCombinateFeatureColStr == "") {
        IsCombinateFeatureCol = ISCOMBINATEFEATURECOL;
    } else {
        IsCombinateFeatureCol = stoi(IsCombinateFeatureColStr);
    }
}

// Free the memory.
RandomForest::~RandomForest() {
    deleteForest();
}

/********************************************************************
 * Description: Train random forest with the data set.
 *
 *@para mat Data set
 *@para featureName The feature name
 *@para argc The first parament of main function
 *@para argv The second parament of main function
 ********************************************************************/
void RandomForest::train(Mat& mat, Row &featureName, int argc, char *argv[]) {
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    if (!isInitialized) {
        MPI_Init(&argc, &argv);
    }
    
    int size, rank;
    MPI_Status  status;
    
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // If need to combinate the feature.
    if (IsCombinateFeatureCol) {
        composIndex = getComposFeatureIndex(featureName, rank);
        composFeature(mat, featureName, composIndex);
    }
    
    int treeTrainNum = 0;
    while (true) {
        // The processor rank == 0 as a controller.
        if( rank == 0 ) {
            
            // When the tree is enough. stop all processor from creating tree.
            if (treeTrainNum >= trainNum)
                treeTrainNum = -1;
            
            for(int i = 1; i < size; i++) {
                if (treeTrainNum != -1)
                    treeTrainNum++;
                
                MPI_Send(&treeTrainNum, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                
                if (treeTrainNum != -1)
                    Info(string("\tTraining the ") + to_string((treeTrainNum)) + string(" tree!"));
            }
            
            if (treeTrainNum == -1)
                break;
            
            int finishTrainTreeNum = 0;
            for(int i = 1; i < size; i++) {
                MPI_Recv(&finishTrainTreeNum, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
            }
        }
        // Other processor as a tree creater.
        else {
            int treeTrainNum;
            MPI_Status  status;
            MPI_Recv(&treeTrainNum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            if (treeTrainNum == -1)
                break;
            
            Mat currentMat;
            Row currentFeatureName;
            this->getDataSet(mat, currentMat, featureName, currentFeatureName,
                             dataScale, featureScale, rank * treeTrainNum);
            DesionTree *trainTree = new DesionTree();
            
            // Create a tree and then store this tree in the processor who created it.
            trainTree->builtTree(currentMat, currentFeatureName);
            trees.push_back(trainTree);
            int size = (int)trees.size();
            
            MPI_Send(&size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }
    }
}

/********************************************************************
 * Description: Predict with random forest.
 *
 *@para mat Data set
 *@para featureName The feature name
 *@para argc The first parament of main function
 *@para argv The second parament of main function
 *
 *@return The predict result
 ********************************************************************/
vector<ElementType> RandomForest::predict(Mat &mat, Row &featureName, int argc, char *argv[]) {
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    if (!isInitialized) {
        MPI_Init(&argc, &argv);
    }
    
    int size, rank;
    MPI_Status  status;
    
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int dataSize = (int)mat.size();
    vector< Row > result;
    
    if (IsCombinateFeatureCol) {
        composFeature(mat, featureName, composIndex);
    }

    if( rank == 0 ) {
        int beginTrain = 1;
        for(int i = 1; i < size; i++) {
            MPI_Send(&beginTrain, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        
        ElementType *resultData = new ElementType[dataSize];
        
        // Get the result from all tree, and then tfind the best one from all the result.
        for(int i = 1; i < size; i++) {
            MPI_Recv(resultData, dataSize, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
            Row row;
            for (int j = 0; j < dataSize; ++j) {
                row.push_back(resultData[j]);
            }
            result.push_back(row);
        }
        delete []resultData;
    }
    else {
        int beginTrain;
        MPI_Status  status;
        MPI_Recv(&beginTrain, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        
        for (auto tree : trees) {
            result.push_back(tree->predict(mat, featureName));
        }
        
        vector<ElementType> resultLabel = getLabelResult(result);
        
        // We can only use mpi api to send array type but not vector.
        // So we should map all result into an array.
        ElementType *resultData = new ElementType[dataSize];
        for (int i = 0; i < resultLabel.size(); ++i) {
            if (i >= dataSize)
                ErrorMesg("\tIndex out of range when map result to array in predict!", true);
            resultData[i] = resultLabel[i];
        }

        // send the predict result to the controller processor to get the final result.
        MPI_Send(resultData, dataSize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        delete []resultData;
    }
    
    if (rank != 0)
        result.clear();
    
    return getLabelResult(result);
}


// Get the result of each row in test data
// In each row, there are TREENUM numbers predict by
//      each tree. This function is to get the one
//      with max times.
vector<ElementType> RandomForest::getLabelResult(vector< Row > &resultList) {
    vector<ElementType> result;
    if (resultList.size() == 0)
        return result;
    
    // labelNumList id to store all the result of label predicted
    // and its times.
    vector<map<ElementType, int> > labelNumList;
    bool flag = true;
    for (auto row : resultList) {
        if (flag) {
            flag = false;
            for (auto item : row) {
                map<ElementType, int> temp;
                temp[item] = 1;
                labelNumList.push_back(temp);
            }
        } else {
            for (int i = 0; i < row.size(); ++i) {
                labelNumList[i][row[i]] += 1;
            }
        }
    }
    
    for (auto labelNum : labelNumList) {
        result.push_back(getResult(labelNum));
    }
    
    return result;
}

// Get the label with max times
ElementType RandomForest::getResult(map<ElementType, int> &labelNum) {
    int maxNum = 0;
    ElementType maxNumLabel = 0.0;
    for (auto item : labelNum) {
        if (maxNum < item.second) {
            maxNum = item.second;
            maxNumLabel = item.first;
        }
    }
    
    return maxNumLabel;
}

// This function is to get a dataset and its feature name for tree train
// To get the data set, you should provide the DATASCALE and FEATURESCALE,
//      which means the scale of the train data for each tree.
// The function chose the train data randomly. The parameter seed is to
// ensure the data set of each tree is different.
void RandomForest::getDataSet(Mat &sourceMat, Mat &aimMat, Row &featureName, Row &aimFeature,
                              double DataSetScale, double featureScale, int seed) {
    if (sourceMat.size() == 0) {
        Info("\tThe data set is empty!");
        return;
    }
    
    aimMat.clear();
    aimFeature.clear();
    
    int featureNum = (int)featureName.size();
    int dataSize = (int)sourceMat.size();
    
    int aimRowNum = (int)(DataSetScale * dataSize);
    int aimFeatureNum = (int)(featureScale * featureNum);
    
    int labelIndex = (int)sourceMat[0].size() - 1;
    
    srand ((unsigned int)time(NULL) * seed);
    set<int> featureIndex;
    
    // The feature in one data set should be different.
    while (featureIndex.size() < aimFeatureNum) {
        int index = rand() % featureNum;
        featureIndex.insert(index);
    }
    
    // The rows in one data set can repeat. which means sampling with replacement
    while (aimMat.size() < aimRowNum) {
        Row rowTemp;
        int rowIndex = rand() % dataSize;
        for (auto featureIndex_  : featureIndex) {
            rowTemp.push_back(sourceMat[rowIndex][featureIndex_]);
        }
        rowTemp.push_back(sourceMat[rowIndex][labelIndex]);
        
        aimMat.push_back(rowTemp);
    }
    
    for (auto featureIndex_  : featureIndex)
        aimFeature.push_back(featureName[featureIndex_]);
}

// Combinate feature in double.
// This function not combinate all feature in composIndex to one
//      but make each double of them into one.
void RandomForest::composFeature(Mat &sourceMat, Row &featureName,
                                 vector<int> &composIndex) {
    int counter = 0;
    while (counter < composIndex.size()) {
        int comIndex1 = composIndex[counter++];
        int comIndex2 = composIndex[counter++];
        
        _composFeature(sourceMat, featureName, comIndex1, comIndex2);
    }
}

// Combinate two feature into one.
void RandomForest::_composFeature(Mat &sourceMat, Row &featureName, int index1, int index2) {
    assert(sourceMat.size() > 0);
    
    int labelIndex = (int)sourceMat[0].size() - 1;
    
    Mat aimMat;
    Row aimFeatureName;
    
    for (int i = 0; i < sourceMat.size(); ++i) {
        Row rowTemp;
        for (int j = 0; j < featureName.size(); ++j) {
            if (j == index1 || j == index2) {
                if (j == index1) {
                    rowTemp.push_back(sourceMat[i][j] COMPOSOPERATION sourceMat[i][index2]);
                }
            } else {
                rowTemp.push_back(sourceMat[i][j]);
            }
        }
        rowTemp.push_back(sourceMat[i][labelIndex]);
        
        aimMat.push_back(rowTemp);
    }
    
    for (int j = 0; j < featureName.size(); ++j) {
        if (j != index2)
            aimFeatureName.push_back(featureName[j]);
    }
    sourceMat = aimMat;
    featureName = aimFeatureName;
}

// Get combinate feature randomly.
// The scale of feature to combinate is the same as the feature
//      one tree get from source data set, which is FEATURESCALE.
vector<int> RandomForest::getComposFeatureIndex(Row &featureName, int seed) {
    vector<int> featureIndex;
    int featureNum = (int)featureName.size();
    int aimFeatureNum = featureNum * featureScale;
    if (aimFeatureNum % 2 != 0)
        aimFeatureNum += 1;
    
    srand((unsigned int)time(NULL) * seed);
    while (featureIndex.size() < aimFeatureNum) {
        int index = rand() % featureNum;
        
        if (find(featureIndex.begin(), featureIndex.end(), index) == featureIndex.end()) {
            featureIndex.push_back(index);
        }
    }
    
    return featureIndex;
}

// Delete trees pointer in forest to free the memory.
void RandomForest::deleteForest() {
    for (int i = 0; i < trees.size(); ++i) {
        trees[i]->deleteTree();
        delete trees[i];
    }
    
    trees.clear();
}