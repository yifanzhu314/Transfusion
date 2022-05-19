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

#include "RawLogReader.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include "torch/script.h"
#include "torch/torch.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include <vector>
#include <chrono>
#include <string>
#include <vector>
#include<fstream>

RawLogReader::RawLogReader(std::string file, bool flipColors)
 : LogReader(file, flipColors)
{
    assert(pangolin::FileExists(file.c_str()));

    fp = fopen(file.c_str(), "rb");
    currentFrame = 0;

    auto tmp = fread(&numFrames,sizeof(int32_t),1,fp);
    assert(tmp);

    depthReadBuffer = new unsigned char[numPixels * 2];
    imageReadBuffer = new unsigned char[numPixels * 3];
    decompressionBufferDepth = new Bytef[Resolution::getInstance().numPixels() * 2];
    decompressionBufferImage = new Bytef[Resolution::getInstance().numPixels() * 3];
    // load_modle();
    // pytorch_module = torch::jit::load("/home/mathloverpi/code/cpp_SINET/cpp/model_cpp.pt");
    // std::ifstream myfile ("/home/mathloverpi/code/ElasticFusion/cpp/model_cpp.pt", std::ifstream::in); 
    pytorch_modle = torch::jit::load("/home/zyf-lab/code_and_data/code/ElastcFusion_Realsense_Camera/cpp/model.pt");
    pytorch_modle.to(torch::kCUDA);                      
    pytorch_modle.eval();
}

    void RawLogReader::load_modle()     
    {          
        // pytorch_modle = torch::jit::load("/home/mathloverpi/code/cpp_SINET/cpp/model_cpp.pt");                
        // pytorch_modle.to(at::kCUDA);    
        // pytorch_module.eval(); 
        ;
                
    } 

RawLogReader::~RawLogReader()
{
    delete [] depthReadBuffer;
    delete [] imageReadBuffer;
    delete [] decompressionBufferDepth;
    delete [] decompressionBufferImage;
    fclose(fp);
}

void RawLogReader::getBack()
{
    assert(filePointers.size() > 0);

    fseek(fp, filePointers.top(), SEEK_SET);

    filePointers.pop();

    getCore();
}

void RawLogReader::getNext()
{
    filePointers.push(ftell(fp));

    getCore();
}

void RawLogReader::getCore()
{
    auto tmp = fread(&timestamp,sizeof(int64_t),1,fp);
    assert(tmp);

    tmp = fread(&depthSize,sizeof(int32_t),1,fp);
    assert(tmp);
    tmp = fread(&imageSize,sizeof(int32_t),1,fp);
    assert(tmp);

    tmp = fread(depthReadBuffer,depthSize,1,fp);
    assert(tmp);

    if(imageSize > 0)
    {
        tmp = fread(imageReadBuffer,imageSize,1,fp);
        assert(tmp);
    }

    if(depthSize == numPixels * 2)
    {
        memcpy(&decompressionBufferDepth[0], depthReadBuffer, numPixels * 2);
    }
    else
    {
        unsigned long decompLength = numPixels * 2;
        uncompress(&decompressionBufferDepth[0], (unsigned long *)&decompLength, (const Bytef *)depthReadBuffer, depthSize);
    }

    if(imageSize == numPixels * 3)
    {
        //std::cout<<1<<std::endl;
        memcpy(&decompressionBufferImage[0], imageReadBuffer, numPixels * 3);
    }
    else if(imageSize > 0)
    {
        jpeg.readData(imageReadBuffer, imageSize, (unsigned char *)&decompressionBufferImage[0]);
    }
    else
    {
        memset(&decompressionBufferImage[0], 0, numPixels * 3);
    }       

depth = (unsigned short *)decompressionBufferDepth;
rgb = (unsigned char *)&decompressionBufferImage[0];


 if(false)
 {
    cv::Mat1b _tmp(480,640);
    cv::Mat3b depthImg(480,640);
    cv::Mat _rgbImg(480,640,CV_8UC3,(cv::Vec<unsigned char, 3> *)rgb);
    cv::Mat rgbImg(480,640,CV_8UC3,(cv::Vec<unsigned char, 3> *)rgb);
    cv::Mat _depth(480,640, CV_16UC1,(unsigned short *)&depth[0]);

    // cv::cvtColor(tmp, depthImg, cv::COLOR_BGR2GRAY);


    cv::Size scale(520, 520);
    cv::resize(_rgbImg, rgbImg, scale, 0, 0, cv::INTER_LINEAR);
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

    // std::cout<<timestamp<<std::endl;

    memcpy(&decompressionBufferDepth[0], (char*)depth_masked.data, numPixels * 2); 
    memcpy(&decompressionBufferImage[0], (char*)rgb_masked.data, numPixels * 3); 
 }
flipColors = 1;
if(flipColors)
{
    for(int i = 0; i < Resolution::getInstance().numPixels() * 3; i += 3)
    {
        std::swap(rgb[i + 0], rgb[i + 2]);
    }
}
    currentFrame++;
}

void RawLogReader::fastForward(int frame)
{
    while(currentFrame < frame && hasMore())
    {
        filePointers.push(ftell(fp));

        auto tmp = fread(&timestamp,sizeof(int64_t),1,fp);
        assert(tmp);

        tmp = fread(&depthSize,sizeof(int32_t),1,fp);
        assert(tmp);
        tmp = fread(&imageSize,sizeof(int32_t),1,fp);
        assert(tmp);

        tmp = fread(depthReadBuffer,depthSize,1,fp);
        assert(tmp);

        if(imageSize > 0)
        {
            tmp = fread(imageReadBuffer,imageSize,1,fp);
            assert(tmp);
        }

        currentFrame++;
    }
}

int RawLogReader::getNumFrames()
{
    return numFrames;
}

bool RawLogReader::hasMore()
{
    return currentFrame + 1 < numFrames;
}


void RawLogReader::rewind()
{
    if (filePointers.size() != 0)
    {
        std::stack<int> empty;
        std::swap(empty, filePointers);
    }

    fclose(fp);
    fp = fopen(file.c_str(), "rb");

    auto tmp = fread(&numFrames,sizeof(int32_t),1,fp);
    assert(tmp);

    currentFrame = 0;
}

bool RawLogReader::rewound()
{
    return filePointers.size() == 0;
}

const std::string RawLogReader::getFile()
{
    return file;
}

void RawLogReader::setAuto(bool value)
{

}
