#pragma once

#include <string>
#include <iostream>
#include <algorithm>
#include <map>
#include <cstring>
#include <cstdio>
#include <atomic>
#include "librealsense2/rs.hpp"

#include "ThreadMutexObject.h"
#include "CameraInterface.h"
#include <iostream>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
// #include<librealsense2/rs.hpp>


class RealSenseInterface : public CameraInterface
{
    public:
        RealSenseInterface(int width = 640,int height = 480,int fps = 30);
        virtual ~RealSenseInterface();

        const int width,height,fps;



        virtual bool ok()
        {
            return initSuccessful;
        }

        virtual std::string error()
        {
            return errorText;
        }

    private:
        rs2::device *dev;
        rs2::context ctx;
        rs2::pipeline pipe;
        std::atomic<bool> pipe_active;

        bool initSuccessful;
        std::string errorText;

        ThreadMutexObject<int> latestRgbIndex;
        std::pair<uint8_t *,int64_t> rgbBuffers[numBuffers];
        // rs2::colorizer c;                          // Helper to colorize depth images

        int64_t lastRgbTime;
        int64_t lastDepthTime;


};
