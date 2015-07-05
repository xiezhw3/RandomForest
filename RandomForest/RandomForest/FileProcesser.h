//
//  FileProcesser.h
//  RandomForest
//
//  Created by wc on 15/6/10.
//  Copyright (c) 2015年 wc. All rights reserved.
//

#ifndef __RandomForest__FileProcesser__
#define __RandomForest__FileProcesser__

#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "Message.h"

using namespace std;

class FileProcesser {
public:
    ~FileProcesser();
    
    vector<vector<string> > readFile(const string &filepath, char delim);
    void writeToFile(vector<vector<string> > &content, const string &filepath, char diam);
    
    static FileProcesser* getInstance();
    
private:
    FileProcesser();
    vector<string> &split(const string &s, char delim, vector<string> &elems);
    vector<string> split(const string &s, char delim);
    
private:
    static FileProcesser* instance;
};

#endif /* defined(__RandomForest__FileProcesser__) */
