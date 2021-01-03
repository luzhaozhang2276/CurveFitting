//
// Created by lu on 2020/12/1.
//

#ifndef DEMO_GLOGHELPER_H
#define DEMO_GLOGHELPER_H


#include <glog/logging.h>
#include <glog/raw_logging.h>

//将信息输出到单独的文件和 LOG(ERROR)
void SignalHandle(const char* data, int size);

class GLogHelper
{
public:
    //GLOG配置：
    GLogHelper(char* program);
    //GLOG内存清理：
    ~GLogHelper();
};


#endif //DEMO_GLOGHELPER_H
