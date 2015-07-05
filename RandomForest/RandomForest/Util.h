//
//  Util.h
//  RandomForest
//
//  Created by wc on 15/6/26.
//  Copyright (c) 2015年 wc. All rights reserved.
//

#ifndef __RandomForest__Util__
#define __RandomForest__Util__

#include <iostream>
#include <string>
#include <map>
#include "FileProcesser.h"
#include "Message.h"

using namespace std;

class Util {
public:
    ~Util();
    
    static Util* getInstance();
    
    string getUtil(const string &key);
    
private:
    Util();
    
private:
    static Util* instance;
    map<string, string> utils;
};

#endif /* defined(__RandomForest__Util__) */
