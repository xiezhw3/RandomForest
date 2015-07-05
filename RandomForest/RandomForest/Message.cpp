//
//  Message.cpp
//  RandomForest
//
//  Created by wc on 15/6/10.
//  Copyright (c) 2015年 wc. All rights reserved.
//

#include "Message.h"

// Show the error message and exit if exit_ is
//      true.
void ErrorMesg(string errorMsg, bool exit_) {
    cout << errorMsg << endl;
    if (exit_)
        exit(0);
}

void DebugMsg(string debugMsg) {
    cout << debugMsg << endl;
}

void Info(string info) {
    cout << info << endl;
}