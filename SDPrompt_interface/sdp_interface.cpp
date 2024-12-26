
#include "sdp_interface.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <fstream>

sdp::SDPrompt_Interface::SDPrompt_Interface(const std::string& vitPath=NULL, const std::string& sdpPath=NULL, const std::string& npyPath=NULL)
{
    if (loadNpy(npyPath, m_all_keys))
    {
		std::cout << "聚类中心npy 加载成功!" << std::endl;
        std::cout << "聚类中心大小: " << m_all_keys.size << std::endl;
        /*for (int i = 0; i < m_all_keys.size[0]; i++)
        {
        	for (int j = 0; j < m_all_keys.size[1]; j++)
        	{
        		for (int k = 0; k < m_all_keys.size[2]; k++)
        		{
        			std::cout << m_all_keys.at<float>(i, j, k) << " ";
        		}
        		std::cout << std::endl;
        	}
        	std::cout << std::endl;
        }*/
	}
    else
    {
		std::cerr << "聚类中心npy 加载失败!" << std::endl;
	}
    if (load(vitPath,m_vit))
    {
		std::cout << "vit 模型加载成功!" << std::endl;
        //printNetLayers(m_vit);
	}
    else
    {
		std::cerr << "vit 模型加载失败!" << std::endl;
	}
    
    if (load(sdpPath, m_sdp))
    {
        std::cout << "sdp 模型加载成功!" << std::endl;
        //printNetLayers(m_vit);
    }
    else
    {
        std::cerr << "sdp 模型加载失败!" << std::endl;
    }


    //printNetLayers(m_sdp);

    //enableCuda(); // 启用cuda
}

void sdp::SDPrompt_Interface::enableCuda()
{
    bool hasCuda = cv::cuda::getCudaEnabledDeviceCount() > 0;

    std::cout << "CUDA 支持情况: " << (hasCuda? "True" : "False") << std::endl;

    if (hasCuda)
    {
        m_cudaEnable = true;
        if (m_vit.net.empty())
        {
			std::cerr << "No model: " << m_vit.path << "CUDA enable fail!" << std::endl;
			return;
		}
		m_vit.net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		m_vit.net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

        if (m_sdp.net.empty())
        {
            std::cerr << "No model: " << m_sdp.path << "CUDA enable fail!" << std::endl;
            return;
        }
        m_sdp.net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        m_sdp.net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

        std::cout << "Run with CUDA! " << std::endl;
    }
    else
    {
        m_cudaEnable = false;
        std::cout << "Run with CPU! " << std::endl;
    }

}

bool sdp::SDPrompt_Interface::load(const std::string& modelPath, sdp::Net& net)
{
    // 判断路径存在
    std::ifstream f(modelPath);
    if (!f)
    {
        std::cerr << "modelPath dose not exist!" << modelPath << std::endl;
        return false;
    }
    net.path = modelPath;

    // 加载模型
    try 
    {
        net.net = cv::dnn::readNetFromONNX(modelPath);
    }
    catch (cv::Exception& e)
    {
        std::cerr << e.what() << std::endl;
		std::cerr << "Failed to load model: " << modelPath << std::endl;
		return false;
	}
    //    m_net = cv::dnn::readNet(m_modelPath);
    if (net.net.empty())
    {
        std::cerr << "Failed to load model: " << net.path << std::endl;
        return false;
    }

    //// 设置 CUDA 设备
    //if (cv::cuda::getCudaEnabledDeviceCount() > 0)
    //{

    //    net.net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    //    net.net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    //    net.useCuda = true;
    //}

    return true;
}

void sdp::SDPrompt_Interface::printNetLayers(sdp::Net& net)
{
    if (net.net.empty()) {
        std::cerr << "No model: " << net.path << std::endl;
        return;
    }

    // 获取网络的层名称
    std::vector<cv::String> layerNames = net.net.getLayerNames();

    std::cout << "Network Layers:" << std::endl;
    for (const auto& layerName : layerNames) {
        // 获取层类型
        std::string layerType = net.net.getLayer(layerName)->type;

        std::cout << "Layer Name: " << layerName << ", Type: " << layerType << std::endl;
    }
}

bool sdp::SDPrompt_Interface::getVitFeature(const cv::Mat& img, cv::Mat& feature)
{
    if (m_vit.net.empty())
    {
		std::cerr << "No model: " << m_vit.path << std::endl;
		return false;
	}
    if (img.empty())
    {
        std::cerr << "img is empty!" << std::endl;
        return false;
    }

    // 图像预处理
    cv::Size imgSize(224, 224);
    cv::Mat blob; // 用于存储预处理后的图像
    if (img.size() != imgSize)
    {
        cv::resize(img, blob, imgSize, 0, 0, cv::INTER_AREA); // 图像缩放
    }
    else
    {
        img.copyTo(blob); 
    }

    //std::cout << blob << std::endl;

    blob = cv::dnn::blobFromImage(blob, 1.0 / 255, blob.size(), cv::Scalar(0, 0, 0), true, false, CV_32F); // 图像转为blob格式 

    //std::cout << blob << std::endl;
    // 输出Blob的内容
    //float* data_ptr = reinterpret_cast<float*>(blob.data);
    //int num_elements = blob.total() * blob.channels();
    //for (int i = 0; i < num_elements; ++i) {
    //    std::cout << data_ptr[i] << " ";
    //}
    //std::cout << std::endl;

    m_vit.net.setInput(blob);


    //std::vector<cv::Mat> outputs;
    //std::vector<std::string> specifiedOutputs = { "logits", "pre_logits" };
    //m_vit.net.forward(outputs, specifiedOutputs); // 获取网络输出


    m_vit.net.forward(feature, "pre_logits"); // 获取网络输出 pre_logits [1,768]
    

    return true;
}

bool sdp::SDPrompt_Interface::getSdpResult(const cv::Mat& img, const cv::Mat& sid, cv::Mat& result)
{
    if (sid.empty())
    {
		std::cerr << "sid is empty!" << std::endl;
		return false;
	}
    if (img.empty())
    {
        std::cerr << "img is empty!" << std::endl;
        return false;
    }
    if (m_sdp.net.empty())
    {
        std::cerr << "No model: " << m_sdp.path << std::endl;
        return false;
    }

    // 图像预处理
    cv::Size imgSize(224, 224);
    cv::Mat blob; // 用于存储预处理后的图像
    if (img.size() != imgSize)
    {
        cv::resize(img, blob, imgSize, 0, 0, cv::INTER_AREA); // 图像缩放
    }
    else
    {
        img.copyTo(blob);
	}
    blob = cv::dnn::blobFromImage(blob, 1/255.0, blob.size(), cv::Scalar(0,0,0), true, false, CV_32F); // 图像转为blob格式
    
    //std::ofstream file("Opencv output.csv");
    //// Access the data in the blob
    //for (int n = 0; n < blob.size[0]; ++n) { // iterate over the batch size
    //    for (int c = 0; c < blob.size[1]; ++c) { // iterate over the channels
    //        float* data = blob.ptr<float>(n, c);
    //        for (int i = 0; i < blob.size[2]; ++i) { // iterate over the height (rows)
    //            for (int j = 0; j < blob.size[3]; ++j) { // iterate over the width (columns)
    //                // Now you can access the pixel value here
    //                // For example, you can print the pixel value
    //                //std::cout << data[i * blob.size[3] + j] << " ";
    //                file << data[i * blob.size[3] + j] << "\n";
    //            }
    //            //std::cout << std::endl;
    //        }
    //    }
    //}
    // 


    try 
    {
        m_sdp.net.setInput(blob, "input");
        m_sdp.net.setInput(sid, "s_id");

        m_sdp.net.forward(result, "logits"); // 获取网络输出
    }
    catch (cv::Exception& e)
    {
		std::cerr << e.what() << std::endl;
		return false;
	}
    return true;
}

void sdp::SDPrompt_Interface::getSid(const cv::Mat& feature, cv::Mat& sid)
{
    cv::Mat resMat = cv::Mat_<float_t>(m_all_keys.size[0],m_all_keys.size[1]); //L1 距离 结果矩阵
    //std::cout << "m_all_keys.size " << m_all_keys.size << std::endl;
    auto all_keys = m_all_keys.reshape(1, 1); // 转为一行

    //std::cout << "m_all_keys.size " << m_all_keys.size << std::endl;
    for (int i = 0; i < m_all_keys.size[0]; ++i)
    {
        for (int j = 0; j < m_all_keys.size[1]; ++j)
        {
            cv::Mat center = all_keys(cv::Rect((i* m_all_keys.size[1] +j) * 768, 0, 768, 1));
            resMat.at<float_t>(i, j) = cv::norm(feature, center, cv::NORM_L1); // 计算L1距
		}
	}

    //显示 resMat 矩阵
    //std::cout << "resMat: " << resMat << std::endl;

    // 获取最小值的索引
    cv::Point minLoc;
    cv::minMaxLoc(resMat, NULL, NULL, &minLoc, NULL);

    //std::cout << "minLoc: " << minLoc << std::endl;

    sid = (cv::Mat_<int32_t>(1, 1) << minLoc.y);
}


bool sdp::SDPrompt_Interface::getSDPromptResult_mat(const cv::Mat& img, cv::Mat& result)
{
    if (m_vit.net.empty())
    {
        std::cerr << "No model: " << m_vit.path << std::endl;
        return false;
    }
    if (img.empty())
    {
        std::cerr << "img is empty!" << std::endl;
        return false;
    }
    if (m_sdp.net.empty())
    {
		std::cerr << "No model: " << m_sdp.path << std::endl;
		return false;
	}

    // 1. 获取特征
    cv::Mat feature;
    if (!getVitFeature(img, feature))
    {
		return false;
	}

    // 2. 获取sid
    cv::Mat sid;
    getSid(feature, sid);

    // 3. 获取结果
    if (!getSdpResult(img, sid, result))
    {
        return false;
    }

    return true;
}

bool sdp::SDPrompt_Interface::getSDPromptResult_int(const cv::Mat& img, int& result)
{
    cv::Mat temp;
    if (!getSDPromptResult_mat(img, temp))
    {
        result = -1;
        return false;
    }


    cv::Point maxLoc;
    cv::minMaxLoc(temp, NULL, NULL, NULL, &maxLoc);

    result = maxLoc.x;

    return true;
}

void sdp::SDPrompt_Interface::warmUp()
{
    cv::Mat dummy = cv::Mat::zeros(224, 224, CV_8UC3);
    cv::Mat temp;
    getVitFeature(dummy, temp);  
    getSdpResult(dummy, cv::Mat::zeros(1,1,CV_8U), temp);
}

bool sdp::SDPrompt_Interface::loadNpy(const std::string& path, cv::Mat& mat)
{
    try 
    {
        cnpy::NpyArray arr = cnpy::npy_load(path);
        float_t* data = arr.data<float_t>();
        std::vector<int>dims;
        for (const int& s : arr.shape)
        {
            dims.push_back(s);
        }
        
        //std::cout << "data " << data << std::endl;
        cv::Mat(dims.size(), &dims[0], CV_32F, data).copyTo(mat); // 深拷贝 先判断 mat是否要重新分配内存
        //mat = cv::Mat(dims.size(), &dims[0], CV_32F, data).clone(); // 深拷贝 后浅拷贝

        //cv::Mat(1, (int)arr.num_vals, CV_32F, data).copyTo(mat);

        //std::cout << "mat  " << reinterpret_cast<void*>(mat.data) << std::endl;
    
    }
    catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return false;
	}

    return true;
}
