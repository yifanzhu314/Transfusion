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

#include "MainController.h"
#include "torch/script.h"
 #include "torch/torch.h"
 #include <vector>
#include <chrono>
#include <string>
#include <iostream>
#include <fstream> 
int main(int argc, char * argv[])
{
    //  torch::DeviceType device_type;
    // device_type = torch::kCUDA;
    // torch::Device device(device_type);
    // std::cout<<"cudu support:"<< (torch::cuda::is_available()?"ture":"false")<<std::endl;
    // // std::ofstream myfile;   
    // myfile.open ("/home/mathloverpi/code/ElasticFusion/cpp/model_cpp.pt");
    // std::ifstream myfile ("/home/mathloverpi/code/ElasticFusion/cpp/model_cpp.pt", std::ifstream::in);
    // torch::jit::script::Module py_module ;
    // py_module = torch::jit::load(myfile);
    // py_module.to(torch::kCUDA);
    MainController mainController(argc, argv);

    mainController.launch();

    return 0;
}
