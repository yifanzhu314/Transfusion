#include "RealSenseInterface.h"
#include <functional>
#include <thread>

#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace cv;


RealSenseInterface::RealSenseInterface(int inWidth,int inHeight,int inFps)
    : width(inWidth),
    height(inHeight),
    fps(inFps),
    dev(nullptr),
    initSuccessful(true)
{
    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_COLOR, inWidth, inHeight, RS2_FORMAT_RGB8,inFps);
    cfg.enable_stream(RS2_STREAM_DEPTH, inWidth, inHeight, RS2_FORMAT_Z16, inFps);




    rs2::device resolve_dev;
    try{
        resolve_dev = cfg.resolve(pipe).get_device();
    }catch(const rs2::error &e){
        initSuccessful = false;
        return;
    }
    dev = &resolve_dev;

    std::cout << "start" << std::endl;
    std::cout << dev->get_info(RS2_CAMERA_INFO_NAME) << " " << dev->get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << std::endl; 

    latestDepthIndex.assign(-1);
    latestRgbIndex.assign(-1);

    for (int i = 0; i < numBuffers; i++){
        uint8_t * newImage = (uint8_t *)calloc(width * height * 3,sizeof(uint8_t));
        rgbBuffers[i] = std::pair<uint8_t *,int64_t>(newImage,0);
    }

    for (int i = 0; i < numBuffers; i++){
        uint8_t * newDepth = (uint8_t *)calloc(width * height * 2,sizeof(uint8_t));
        uint8_t * newImage = (uint8_t *)calloc(width * height * 3,sizeof(uint8_t));
        frameBuffers[i] = std::pair<std::pair<uint8_t *,uint8_t *>,int64_t>(std::pair<uint8_t *,uint8_t *>(newDepth,newImage),0);
    }



	rs2::frame_queue queue(numBuffers);
    std::thread t([&,cfg, inWidth, inHeight, inFps]() {
        rs2_stream align_to = RS2_STREAM_COLOR;
        rs2::align align(align_to);
        rs2::pipeline_profile profile = pipe.start(cfg);
        rs2::colorizer c;                          // Helper to colorize depth images

        pipe_active = true;

        while (pipe_active)
        {
            auto frameset = pipe.wait_for_frames();
        

            auto processed = align.process(frameset);

            rs2::video_frame color_frame = processed.get_color_frame();//first(RS2_STREAM_COLOR);
            rs2::depth_frame depth_frame = processed.get_depth_frame();


        //     rs2::frame aligned_depth_frame = processed.get_depth_frame().apply_filter(c);;
 
        // //获取对齐之前的color图像
        // //获取宽高
        // const int depth_w=aligned_depth_frame.as<rs2::video_frame>().get_width();
        // const int depth_h=aligned_depth_frame.as<rs2::video_frame>().get_height();
        // const int color_w=color_frame.as<rs2::video_frame>().get_width();
        // const int color_h=color_frame.as<rs2::video_frame>().get_height();

        // //If one of them is unavailable, continue iteration

        // // //创建OPENCV类型 并传入数据
        // Mat aligned_depth_image(Size(depth_w,depth_h),CV_8UC3,(void*)aligned_depth_frame.get_data(),Mat::AUTO_STEP);
        // Mat aligned_color_image(Size(color_w,color_h),CV_8UC3,(void*)color_frame.get_data(),Mat::AUTO_STEP);
        // //显示
        // imshow("depth_win",aligned_depth_image);
        // imshow("color_win",aligned_color_image);
        // waitKey(1);






            if (!depth_frame)
            {
                continue;
            }

            lastDepthTime = lastRgbTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
            int bufferIndex = (latestRgbIndex.getValue() + 1) % numBuffers;
            //RGB

            memcpy(rgbBuffers[bufferIndex].first,color_frame.get_data(),
                    color_frame.get_width() * color_frame.get_height() * 3);

            rgbBuffers[bufferIndex].second = lastRgbTime;
            latestRgbIndex++;


            //Depth
            // The multiplication by 2 is here because the depth is actually uint16_t
            memcpy(frameBuffers[bufferIndex].first.first,depth_frame.get_data(),
                    depth_frame.get_width() * depth_frame.get_height() * 2);

            frameBuffers[bufferIndex].second = lastDepthTime;

            int lastImageVal = latestRgbIndex.getValue();

            if(lastImageVal == -1)
            {
                return;
            }

            lastImageVal %= numBuffers;

            memcpy(frameBuffers[bufferIndex].first.second,rgbBuffers[lastImageVal].first,
                    depth_frame.get_width() * depth_frame.get_height() * 3);

            latestDepthIndex++;

        }
        pipe.stop();
    });
    t.detach();
}

RealSenseInterface::~RealSenseInterface()
{
    if(initSuccessful)
    {
        pipe_active = false;

        for(int i = 0; i < numBuffers; i++)
        {
            free(rgbBuffers[i].first);
        }

        for(int i = 0; i < numBuffers; i++)
        {
            free(frameBuffers[i].first.first);
            free(frameBuffers[i].first.second);
        }

    }
}







