//
//  Test.cpp
//  RandomForest
//
//  Created by wc on 15/6/11.
//  Copyright (c) 2015年 wc. All rights reserved.
//

#include "Run.h"


// Init mpi and get the processors number of this program.
// This function also get the rank of the processor.
void initMpiAndGetInfo(int argc, char *argv[], int &rank, int &size) {
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    if (!isInitialized) {
        MPI_Init(&argc, &argv);
    }
    // Get the number of processors
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Get the rank of this procressor.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}

// This function use cv data to make a test of the program.
// In this function, I use MPI API to make my program
//      parallelize.
void test(int argc, char *argv[]) {
    int rank, size;
    initMpiAndGetInfo(argc, argv, rank, size);
    MPI_Status  status;
    
    vector< vector<string> > trainMat;
    Row featureName;
    Mat dataMat;
    readDate(argc, argv, CVTESTTRAINFILE, trainMat);
    getMatAndFeature(trainMat, true, dataMat, featureName);
    
    // All procressors run into train function.
    RandomForest forest;
    forest.train(dataMat, featureName, argc, argv);
    
    vector<int> labelData;
    vector< vector<string> > testMatCV;
    Mat testDataMat;
    Row testFeatureNameCV;
    
    readDate(argc, argv, CVTESTTESTFILE, testMatCV);
    labelData = getCVData(testMatCV, testDataMat, testFeatureNameCV);
    
    // Each procressor predict with test data independently use the tree it build itself.
    // After each procressor finish predict, procressors from 1 to max should send
    //      the predict result to procressor 0. And the procressor 0 will get the best result.
    vector<ElementType> result = forest.predict(testDataMat, testFeatureNameCV, argc, argv);
    
    // This part is to make sure all procressors have finished its work before exit.
    if( rank == 0 ) {
        Info("Finish predicting, calculate the error rate...");
        int errorCount = 0;
        for (int i = 0; i != result.size(); ++i) {
            if (getIntValue(result[i]) != labelData[i])
                errorCount += 1;
        }
        
        Info(string("Error Rate: ") + to_string(errorCount / (float)(result.size())));
        
        int dataOut = 1;
        for(int pr = 1; pr < size; pr++) {
            MPI_Send(&dataOut, 1, MPI_INT, pr, 0, MPI_COMM_WORLD);
        }
    }
    else {
        int message;
        MPI_Recv(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }
}


// This code is to get the result of test file.
// The test file is who has data set with feature
//      but not label, and this code is to get
//      the label with feature.
void runMain(int argc, char *argv[]) {
    int rank, size;
    initMpiAndGetInfo(argc, argv, rank, size);
    MPI_Status  status;
    
    vector< vector<string> > trainMat;
    Row featureName;
    Mat dataMat;
    readDate(argc, argv, TRAINFILE, trainMat);
    getMatAndFeature(trainMat, true, dataMat, featureName);
    
    // Begin to train the random forest.
    RandomForest forest;
    forest.train(dataMat, featureName, argc, argv);
    
    vector< vector<string> > testMat;
    Mat testDataMat;
    Row testFeatureName;
    
    readDate(argc, argv, TESTFILE, testMat);
    getMatAndFeature(testMat, false, testDataMat, testFeatureName);
    
    // Each procressor predict with test data independently use the tree it build itself.
    // After each procressor finish predict, procressors from 1 to max should send
    //      the predict result to procressor 0. And the procressor 0 will get the best result.
    vector<ElementType> result = forest.predict(testDataMat, testFeatureName, argc, argv);
    
    // This part is to make sure all procressors have finished its work before exit.
    if( rank == 0 ) {
        Info("Finish predicting, writting the result to file...");
        vector< vector<string> > resultToFile;
        vector<string> featureNameToFile;
        featureNameToFile.push_back("id,label");
        resultToFile.push_back(featureNameToFile);
        
        int index = 0;
        for (auto item : result) {
            vector<string> row;
            row.push_back(to_string(index++));
            row.push_back(to_string(getIntValue(item)));
            resultToFile.push_back(row);
        }
        
        // Write result to file.
        FileProcesser::getInstance()->writeToFile(resultToFile, RESULTFILE, DELIM);
        
        Info(string("Finish writting the result to path <") + string(RESULTFILE) + string(">"));
        
        int dataOut = 1;
        for(int pr = 1; pr < size; pr++) {
            MPI_Send(&dataOut, 1, MPI_INT, pr, 0, MPI_COMM_WORLD);
        }
    }
    else {
        int message;
        MPI_Recv(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }
}

// This function is to chose feaature.
// This function tey to change the secquence
//      of each feature to find if the feature
//      make greate influence of the model,
//      if not, delete the feature.
void choseFeature(int argc, char *argv[]) {
    int rank, size;
    initMpiAndGetInfo(argc, argv, rank, size);
    MPI_Status  status;
    
    vector< vector<string> > trainMat;
    Row featureName;
    Mat dataMat;
    
    readDate(argc, argv, CVTESTTRAINFILE, trainMat);
    getMatAndFeature(trainMat, true, dataMat, featureName);
    
    // All procressors run into train function.
    RandomForest forest;
    forest.train(dataMat, featureName, argc, argv);
    
    vector<int> labelData;
    vector< vector<string> > testMatCV;
    Mat testDataMat;
    Row testFeatureNameCV;
    
    readDate(argc, argv, CVTESTTESTFILE, testMatCV);
    labelData = getCVData(testMatCV, testDataMat, testFeatureNameCV);
    
    // Each procressor predict with test data independently use the tree it build itself.
    // After each procressor finish predict, procressors from 1 to max should send
    //      the predict result to procressor 0. And the procressor 0 will get the best result.
    vector<ElementType> result = forest.predict(testDataMat, testFeatureNameCV, argc, argv);
    
    // This part is to make sure all procressors have finished its work before exit.
    double errorRate = 0.0;
    if( rank == 0 ) {
        Info("Finish predicting, calculate the error rate...");
        int errorCount = 0;
        for (int i = 0; i != result.size(); ++i) {
            if (getIntValue(result[i]) != labelData[i])
                errorCount += 1;
        }
        
        errorRate = errorCount / (float)(result.size());
        Info(string("Error Rate: ") + to_string(errorRate));
        
        int dataOut = 1;
        for(int pr = 1; pr < size; pr++) {
            MPI_Send(&dataOut, 1, MPI_INT, pr, 0, MPI_COMM_WORLD);
        }
    }
    else {
        int message;
        MPI_Recv(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }

    int featureSize = (int)testFeatureNameCV.size();
    double *errorRateDiff = new double[featureSize];
    
    for (int i = 0; i < featureSize; ++i) {
        Mat trainataTemp = dataMat;
        disorganizeFeature(trainataTemp, i);
        
        RandomForest forestTemp;
        forestTemp.train(trainataTemp, featureName, argc, argv);
        
        if( rank == 0 ) {
            int dataOut = 1;
            for(int pr = 1; pr < size; pr++) {
                MPI_Send(&dataOut, 1, MPI_INT, pr, 0, MPI_COMM_WORLD);
            }
        }
        else {
            int message;
            MPI_Recv(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        }
        
        vector<ElementType> result = forestTemp.predict(testDataMat, testFeatureNameCV, argc, argv);
        
        if( rank == 0 ) {
            int errorCount = 0;
            for (int i = 0; i != result.size(); ++i) {
                if (getIntValue(result[i]) != labelData[i])
                    errorCount += 1;
            }
            double errorRateTemp = errorCount / (double)(result.size());
            errorRateDiff[i] = fabs(errorRateTemp - errorRate);
            
            Info(string("Error Rate difference of feature ") + to_string(i) + string(" is: ") +
                 to_string(errorRateTemp - errorRate));
            
            int dataOut = 1;
            for(int pr = 1; pr < size; pr++) {
                MPI_Send(&dataOut, 1, MPI_INT, pr, 0, MPI_COMM_WORLD);
            }
        }
        else {
            int message;
            MPI_Recv(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        }
    }
    
    if( rank == 0 ) {
        for (int i = 0; i < featureSize; ++i) {
            Info(string("Error rate difference of feature ") + to_string(i) +
                 string(" and the standard is ") + to_string(errorRateDiff[i]));
        }
    }

    if( rank == 0 ) {
        for(int pr = 1; pr < size; pr++) {
            MPI_Send(errorRateDiff, featureSize, MPI_DOUBLE, pr, 0, MPI_COMM_WORLD);
        }
    }
    else {
        MPI_Recv(errorRateDiff, featureSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    }
    
    multimap<double, int> errorRateIndex;
    for (int i = 0; i < featureSize; ++i) {
        errorRateIndex.insert(pair<double, int>(errorRateDiff[i], i));
    }
    
    vector<int> featureDeleteIndexList = getFeatureDeleteIndex(errorRateIndex, FEATURETHRESHOD);
    
    int dataIn = 0, dataOut = 0;
    if( rank == 0 ) {
        for(int pr = 1; pr < size; pr++) {
            MPI_Send(&dataOut, 1, MPI_INT, pr, 0, MPI_COMM_WORLD);
        }
    }
    else {
        MPI_Recv(&dataIn, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }
    
    testWithDeleteFeature(argc, argv, featureDeleteIndexList);
    
    if( rank == 0 ) {
        for(int pr = 1; pr < size; pr++) {
            MPI_Send(&dataOut, 1, MPI_INT, pr, 0, MPI_COMM_WORLD);
        }
    }
    else {
        MPI_Recv(&dataIn, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }
    
    runMainWithDeleteFeature(argc, argv, featureDeleteIndexList);
    
}

// This function is to test whether the effect of the model with deleted
//      is better then source data.
void runMainWithDeleteFeature(int argc, char *argv[], vector<int> &featureDeleteIndexList) {
    int rank, size;
    initMpiAndGetInfo(argc, argv, rank, size);
    MPI_Status  status;
    
    vector< vector<string> > trainMat;
    Row featureName;
    Mat dataMat;
    readDate(argc, argv, TRAINFILE, trainMat);
    getMatAndFeature(trainMat, true, dataMat, featureName);
    deleteFeature(dataMat, featureName, featureDeleteIndexList);
    
    // Begin to train the random forest.
    RandomForest forest;
    forest.train(dataMat, featureName, argc, argv);
    
    vector<int> labelData;
    vector< vector<string> > testMat;
    Mat testDataMat;
    Row testFeatureName;
    
    readDate(argc, argv, TESTFILE, testMat);
    getMatAndFeature(testMat, false, testDataMat, testFeatureName);
    deleteFeature(testDataMat, testFeatureName, featureDeleteIndexList);
    
    // Each procressor predict with test data independently use the tree it build itself.
    // After each procressor finish predict, procressors from 1 to max should send
    //      the predict result to procressor 0. And the procressor 0 will get the best result.
    vector<ElementType> result = forest.predict(testDataMat, testFeatureName, argc, argv);
    
    // This part is to make sure all procressors have finished its work before exit.
    if( rank == 0 ) {
        Info("Finish predicting, writting the result to file...");
        vector< vector<string> > resultToFile;
        vector<string> row_;
        row_.push_back("id,label");
        resultToFile.push_back(row_);
        
        int index = 0;
        for (auto item : result) {
            vector<string> row;
            row.push_back(to_string(index++));
            row.push_back(to_string(getIntValue(item)));
            resultToFile.push_back(row);
        }
        
        // Write result to file.
        FileProcesser::getInstance()->writeToFile(resultToFile, RESULTFILE, DELIM);
        
        Info(string("Finish writting the result to path <") + string(RESULTFILE) + string(">"));
        
        int dataOut = 1;
        for(int pr = 1; pr < size; pr++) {
            MPI_Send(&dataOut, 1, MPI_INT, pr, 0, MPI_COMM_WORLD);
        }
    }
    else {
        int message;
        MPI_Recv(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }
}

void testWithDeleteFeature(int argc, char *argv[], vector<int> &featureDeleteIndexList) {
    int rank, size;
    initMpiAndGetInfo(argc, argv, rank, size);
    MPI_Status  status;
    
    vector< vector<string> > trainMat;
    Row featureName;
    Mat dataMat;
    readDate(argc, argv, CVTESTTRAINFILE, trainMat);
    getMatAndFeature(trainMat, true, dataMat, featureName);
    deleteFeature(dataMat, featureName, featureDeleteIndexList);
    
    // All procressors run into train function.
    RandomForest forest;
    forest.train(dataMat, featureName, argc, argv);
    
    vector<int> labelData;
    vector< vector<string> > testMatCV;
    Mat testDataMat;
    Row testFeatureNameCV;
    
    readDate(argc, argv, CVTESTTESTFILE, testMatCV);
    labelData = getCVData(testMatCV, testDataMat, testFeatureNameCV);
    deleteFeature(testDataMat, testFeatureNameCV, featureDeleteIndexList);
    
    // Each procressor predict with test data independently use the tree it build itself.
    // After each procressor finish predict, procressors from 1 to max should send
    //      the predict result to procressor 0. And the procressor 0 will get the best result.
    vector<ElementType> result = forest.predict(testDataMat, testFeatureNameCV, argc, argv);
    
    // This part is to make sure all procressors have finished its work before exit.
    if( rank == 0 ) {
        Info("Finish predicting, calculate the error rate...");
        int errorCount = 0;
        for (int i = 0; i != result.size(); ++i) {
            if (getIntValue(result[i]) != labelData[i])
                errorCount += 1;
        }
        
        Info(string("Error Rate: ") + to_string(errorCount / (float)(result.size())));
        
        int dataOut = 1;
        for(int pr = 1; pr < size; pr++) {
            MPI_Send(&dataOut, 1, MPI_INT, pr, 0, MPI_COMM_WORLD);
        }
    }
    else {
        int message;
        MPI_Recv(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }
}

// This function is to get data set of each processor paiallelize.
// In this function, the program has to ensure all the processor
//      has finish get data from file before one of the processor
//      return.
void readDate(int argc, char *argv[], const string &filepath,
              vector< vector<string> > &mat) {
    mat.clear();

    int rank, size;
    initMpiAndGetInfo(argc, argv, rank, size);
    MPI_Status  status;
    
    int dataOut = 0;
    
    while (true) {
        // Make procressor 0 a controller.
        if( rank == 0 ) {
            Info("Getting data from file...");
            
            int dataIn = 0, pr;
            for(pr = 1; pr < size; pr++) {
                MPI_Send(&dataOut, 1, MPI_INT, pr, 0, MPI_COMM_WORLD);
            }
            
            if (mat.size() == 0) {
                mat = FileProcesser::getInstance()->readFile(filepath, DELIM);
            }
            
            // This means all procressors have finished read file.
            if (dataOut == 1) {
                break;
            }
            
            int num = 0;
            for(pr = 1; pr < size; pr++) {
                MPI_Recv(&dataIn, 1, MPI_INT, pr, 1, MPI_COMM_WORLD, &status);
                if (dataIn == 1)
                    ++num;
            }
            // Have received symbol from all procressors.
            if (num == size - 1) {
                dataOut = 1;
            }
        }
        else {
            int message;
            MPI_Status  status;
            MPI_Recv(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            
            if (message == 1)
                break;
            
            // If receive begin symbol from procressor 0, begin
            //      to read file. And send symbol to procressor
            //      0 after finish reading.
            if (mat.size() == 0) {
                mat = FileProcesser::getInstance()->readFile(filepath, DELIM);
            }
            
            int dataOutTemp = 1;
            MPI_Send(&dataOutTemp, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }
    }
}

// Find the feature whose error rate is close to standard error rate
//      after change the feature secquence.
vector<int> getFeatureDeleteIndex(multimap<double, int> &errorRateIndex,
                                  double threshod) {
    vector<int> featureDeleteIndexList;
    for (auto item : errorRateIndex) {
        if (item.first < FEATURETHRESHOD) {
            featureDeleteIndexList.push_back(item.second);
        }
    }
    return featureDeleteIndexList;
}