//
//  main.cpp
//  RandomForest
//
//  Created by wc on 15/6/10.
//  Copyright (c) 2015年 wc. All rights reserved.
//

#include <iostream>
#include "FileProcesser.h"
#include "DesionTree.h"
#include "RandomForest.h"
#include "tools.h"
#include "Run.h"
#include "Util.h"

// If the TEST paramenter is set whether need to make test.
#define CVTEST false

// If need to run the test file to predict the label, change
//      this to true.
#define PREDICTTEST false

// Feature chosen. Run this part to find the feature who
//      make little influence to the model and delete it
//      from the data set.
#define CHOSEFEATUREBEFORERUN false

int main(int argc, char * argv[]) {
    bool cvTest, predictTest, choseFeatureBeforeRun;
    
    string cvTestStr = Util::getInstance()->getUtil("CVTEST");
    if (cvTestStr == "") {
        cvTest = CVTEST;
    } else {
        cvTest = stoi(cvTestStr);
    }
    
    string predictTestStr = Util::getInstance()->getUtil("PREDICTTEST");
    if (predictTestStr == "") {
        predictTest = PREDICTTEST;
    } else {
        predictTest = stoi(predictTestStr);
    }
    
    string choseFeatureBeforeRunStr = Util::getInstance()->getUtil("CHOSEFEATUREBEFORERUN");
    if (choseFeatureBeforeRunStr == "") {
        choseFeatureBeforeRun = CHOSEFEATUREBEFORERUN;
    } else {
        choseFeatureBeforeRun = stoi(choseFeatureBeforeRunStr);
    }
    
    //-------------- Do cv test of the random forest -----------------
    if(cvTest) {
        test(argc, argv);
    }
    //----------------------------------------------------------------
    
    
    // ------------- Run test to get the result ----------------------
    if (predictTest) {
        runMain(argc, argv);
    }
    //----------------------------------------------------------------
    
    
    // ------ Chose the feature to train before run the test file ----
    if (choseFeatureBeforeRun) {
        choseFeature(argc, argv);
    }
    //----------------------------------------------------------------
    
    return 0;
}
