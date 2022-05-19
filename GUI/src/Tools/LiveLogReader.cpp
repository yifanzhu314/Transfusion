/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#include "LiveLogReader.h"
// #include<opencv/core.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui_c.h> 
#include "RealSenseInterface.h"
#include <string>

LiveLogReader::LiveLogReader(std::string file, bool flipColors, CameraType type)
 : LogReader(file, flipColors),
   lastFrameTime(-1),
   lastGot(-1)
{
    std::cout << "Creating live capture... "; std::cout.flush();


    cam = new RealSenseInterface(Resolution::getInstance().width(), Resolution::getInstance().height());


	decompressionBufferDepth = new Bytef[Resolution::getInstance().numPixels() * 2];

	decompressionBufferImage = new Bytef[Resolution::getInstance().numPixels() * 3];
    pytorch_modle = torch::jit::load("/home/zyf-lab/code_and_data/code/ElastcFusion_Realsense_Camera/cpp/model.pt");
    pytorch_modle.to(torch::kCUDA);  
    pytorch_modle.eval();
    if(!cam || !cam->ok())
    {
        std::cout << "failed!" << std::endl;
        std::cout << cam->error();
    }
    else
    {
        std::cout << "success!" << std::endl;

        std::cout << "Waiting for first frame"; std::cout.flush();

        int lastDepth = cam->latestDepthIndex.getValue();

        do
        {
          std::this_thread::sleep_for(std::chrono::microseconds(33333));
            std::cout << "."; std::cout.flush();
            lastDepth = cam->latestDepthIndex.getValue();
        } while(lastDepth == -1);

        std::cout << " got it!" << std::endl;
    }
}

LiveLogReader::~LiveLogReader()
{
    delete [] decompressionBufferDepth;

    delete [] decompressionBufferImage;

	delete cam;
}

void LiveLogReader::getNext()
{
    int lastDepth = cam->latestDepthIndex.getValue();

    assert(lastDepth != -1);

    int bufferIndex = lastDepth % CameraInterface::numBuffers;

    if(bufferIndex == lastGot)
    {
        return;
    }

    if(lastFrameTime == cam->frameBuffers[bufferIndex].second)
    {
        return;
    }

    memcpy(&decompressionBufferDepth[0], cam->frameBuffers[bufferIndex].first.first, Resolution::getInstance().numPixels() * 2);
    memcpy(&decompressionBufferImage[0], cam->frameBuffers[bufferIndex].first.second,Resolution::getInstance().numPixels() * 3);

    lastFrameTime = cam->frameBuffers[bufferIndex].second;

    timestamp = lastFrameTime;

    rgb = (unsigned char *)&decompressionBufferImage[0];
    depth = (unsigned short *)&decompressionBufferDepth[0];


        cv::Mat1b _tmp(480,640);
        cv::Mat3b depthImg(480,640);
        cv::Mat _rgbImg(480,640,CV_8UC3,(cv::Vec<unsigned char, 3> *)rgb);
        cv::Mat rgbImg(480,640,CV_8UC3,(cv::Vec<unsigned char, 3> *)rgb);
        cv::Mat _depth(480,640, CV_16UC1,(unsigned short *)&depth[0]);
        cv::Size scale(520, 520);
        cv::resize(_rgbImg, rgbImg, scale, 0, 0, cv::INTER_LINEAR);
        // cv::cvtColor(tmp, depthImg, cv::COLOR_BGR2GRAY);
        cv::cvtColor(_rgbImg, _rgbImg, cv::COLOR_RGB2BGR);


        std::stringstream rgb_strs;
        std::stringstream depth_strs;
        rgb_strs << "/home/zyf-lab/code_and_data/code/ElastcFusion_Realsense_Camera/data/live_data/rgb/"<<std::setprecision(6) << std::fixed << double(timestamp)/1000.0 << ".png";

        depth_strs << "/home/zyf-lab/code_and_data/code/ElastcFusion_Realsense_Camera/data/live_data/depth/"<<std::setprecision(6) << std::fixed << double(timestamp)/1000.0 << ".png";

        cv::imwrite(rgb_strs.str(),_rgbImg);
        cv::imwrite(depth_strs.str(),_depth);

        // _depth = _depth*(1000/100);

        // memcpy(&decompressionBufferDepth[0], (char*)_depth.data, numPixels * 2); 
  if(false){

        //transforms.ToTensor()
        rgbImg.convertTo(rgbImg, CV_32FC3, 1.0 / 255.0);
        torch::Tensor tensor_image = torch::from_blob(rgbImg.data, {1, rgbImg.rows, rgbImg.cols,3});
        tensor_image = tensor_image.permute({0,3,1,2});

        //transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        tensor_image[0][0] = tensor_image[0][0].sub_(0.485).div_(0.229);
        tensor_image[0][1] = tensor_image[0][1].sub_(0.456).div_(0.224);
        tensor_image[0][2] = tensor_image[0][2].sub_(0.406).div_(0.225);
        tensor_image = tensor_image.to(torch::kCUDA);

        std::vector<at::Tensor> dataOutputAll = pytorch_modle.forward({tensor_image}).toTensorVector(); 
        torch::Tensor result = dataOutputAll[0];
        result = result.argmax(1)[0];

        result = result.mul(127).to(torch::kU8) ;
        result = result.to(torch::kCPU);
        cv::Mat pts_mat(cv::Size(520, 520), CV_8U, result.data_ptr());
        resize(pts_mat, pts_mat, cv::Size(640, 480)); 
        // threshold(pts_mat, pts_mat, 128, 255, cv::THRESH_TOZERO_INV);
        threshold(pts_mat, pts_mat, 126, 255, cv::THRESH_BINARY);
        pts_mat = ~ pts_mat; 
        cv::Mat depth_masked;
        cv::Mat rgb_masked;

        _depth.copyTo(depth_masked, pts_mat); 
        _rgbImg.copyTo(rgb_masked,pts_mat);
        depth_masked = depth_masked*(1000/100);

        memcpy(&decompressionBufferDepth[0], (char*)depth_masked.data, numPixels * 2); 
        memcpy(&decompressionBufferImage[0], (char*)rgb_masked.data, numPixels * 3); 
    }
        // cv::imshow("depth_win",_depth);
        // cv::imshow("color_win",_rgbImg);
        // cv::waitKey(1);


        // for(int i = 0; i < Resolution::getInstance().numPixels() * 3; i += 3)
        // {
        //     std::swap(rgb[i + 0], rgb[i + 2]);
        // }




        // std::cout<<rgb_strs.str()<<std::endl;















        // depth = (unsigned short *)decompressionBufferDepth;

        // cv::imshow("RGB", rgbImg);
        
        // cv::imshow("Depth", _depth);
        
        // char key = cv::waitKey(1);





    imageReadBuffer = 0;
    depthReadBuffer = 0;

    imageSize = Resolution::getInstance().numPixels() * 3;
    depthSize = Resolution::getInstance().numPixels() * 2;

    if(flipColors)
    {
        for(int i = 0; i < Resolution::getInstance().numPixels() * 3; i += 3)
        {
            std::swap(rgb[i + 0], rgb[i + 2]);
        }
    }
}

const std::string LiveLogReader::getFile()
{
    return Parse::get().baseDir().append("live");
}

int LiveLogReader::getNumFrames()
{
    return std::numeric_limits<int>::max();
}

bool LiveLogReader::hasMore()
{
    return true;
}

void LiveLogReader::setAuto(bool value)
{
    // cam->setAutoExposure(value);
    // cam->setAutoWhiteBalance(value);
    ;
}
