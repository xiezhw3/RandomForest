//
//  FileProcesser.cpp
//  RandomForest
//
//  Created by wc on 15/6/10.
//  Copyright (c) 2015年 wc. All rights reserved.
//

#include "FileProcesser.h"

FileProcesser* FileProcesser::instance = nullptr;

FileProcesser::~FileProcesser() {
    if (instance != nullptr)
        delete instance;
}

FileProcesser::FileProcesser() { }

// Use singleton nodle.
FileProcesser* FileProcesser::getInstance() {
    if (instance == nullptr)
        instance = new FileProcesser();
    return instance;
}

// This function is to read file, and split each row with the splitSimble
//      provided.
vector< vector<string> > FileProcesser::readFile(const string &filePath,
                                                 char splitSimble) {
    //Info(string("Begin to read file ") + filePath);
    vector< vector<string> > result;
    ifstream in(filePath);
    char buff[12401];
    if (in.is_open()) {
        while (!in.eof()) {
            in.getline(buff, 12400);
            result.push_back(split(buff, splitSimble));
        }
    } else {
        ErrorMesg(string("Open file ") + filePath + string(" fail!"), true);
    }
    //Info(string("Finish read file ") + filePath);
    
    return result;
}

// Write the data to file. Use splitSimble between each element.
void FileProcesser::writeToFile(vector< vector<string> > &result,
                                const string &filePath, char splitSimble) {
    //Info(string("Begin to read file ") + filePath);
    
    ofstream out(filePath, std::ofstream::out);
    for (int i = 0; i < result.size(); ++i) {
        for (int j = 0; j < result[i].size(); ++j) {
            if (j == 0) {
                out << result[i][j];
            }
            else
                out << splitSimble << result[i][j];
        }
        out << endl;
    }
    out.close();
    
    //Info(string("Finish read file ") + filePath);
}

// Split the string with delim.
vector< string>& FileProcesser::split(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    
    return elems;
}

vector<string> FileProcesser::split(const string &s, char delim) {
    vector<string> elems;
    split(s, delim, elems);
    
    return elems;
}