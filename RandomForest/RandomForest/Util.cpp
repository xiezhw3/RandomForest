//
//  Util.cpp
//  RandomForest
//
//  Created by wc on 15/6/26.
//  Copyright (c) 2015年 wc. All rights reserved.
//

#include "Util.h"

Util* Util::instance = nullptr;

// Get the util message from util file.
Util::Util() {
    vector< vector<string> > result = FileProcesser::getInstance()->readFile("./Util/util", ':');
    for (auto line : result) {
        if (line.size() == 2)
            utils[line[0]] = line[1];
    }
}

Util::~Util() {
    if (instance != nullptr)
        delete instance;
}

Util* Util::getInstance() {
    if (instance == nullptr)
        instance = new Util();
    return instance;
}

// Get the util value by key.
string Util::getUtil(const string &key) {
    if (utils.find(key) != utils.end())
        return utils[key];
    return "";
}