//#include <onnxruntime/core/providers/cuda/cuda_provider_factory.h>
//#include <onnxruntime_cxx_api.h>

#include <iostream>
#include "opencv2/opencv.hpp"

#include "sdp_ort_interface.h"

int main()
{

    sdp::sdp_ort_interface sdp(L"sdp_17.onnx", L"vit_sid.onnx");
    int res;
    cv::Mat mat = cv::imread("E:/师兄代码/已归档数据集/myDataset/mya1/train/fake/1100.bmp");
    //sdp.warmUp();
    // 计算程序运行时间
   // 获取程序开始时间点

    for (int i = 0; i < 100; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        sdp.infer(mat, res);

        // 获取程序结束时间点
        auto end = std::chrono::high_resolution_clock::now();

        // 计算程序运行时间，以毫秒为单位
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // 输出运行时间
        std::cout << "程序运行时间为：" << duration.count() << " 毫秒\n";
        std::cout << "res: " << res << std::endl;

        Sleep(500);
    }

    //// 初始化ONNX Runtime
    //Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sdp infer");

    //// 创建会话选项并添加CUDA提供程序
    //Ort::SessionOptions session_options;
    //uint32_t device_id = 0; // CUDA 设备 ID
    ////Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, device_id));
    //Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, device_id));

    //// 加载模型
    //const std::wstring model_path = L"sdp_17.onnx";
    //Ort::Session session(env, model_path.c_str(), session_options);
    //    



   
    //// 用于保存input张量值
    //std::vector<float> input_tensor_values;

    //// 创建输入张量
    //cv::Mat img = cv::imread("E:/师兄代码/已归档数据集/myDataset/mya1/train/fake/1100.bmp");
    //cv::Size imgSize(224, 224); 
    //cv::Mat blob; // 用于存储预处理后的图像
    //if (img.size() != imgSize) 
    //{
    //    cv::resize(img, blob, imgSize, 0, 0, cv::INTER_AREA); // 图像缩放 
    //}
    //else
    //{
    //    img.copyTo(blob); 
    //}

    //blob = cv::dnn::blobFromImage(blob, 1.0 / 255, blob.size(), cv::Scalar(0, 0, 0), true, false, CV_32F); // 图像转为blob格式 

    //float* data_ptr = reinterpret_cast<float*>(blob.data); // 获取blob数据指针
    //int num_elements = blob.total() * blob.channels(); // 获取blob数据元素个数
    //// 将blob数据拷贝到input张量中
    //for (int i = 0; i < num_elements; ++i) 
    //{
    //    /*std::cout << data_ptr[i] << " ";*/
    //    input_tensor_values.push_back(std::move(data_ptr[i]));
    //}

    //// 创建内存信息
    //auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    //
    //// 创建input张量
    //std::vector<int64_t> input_tensor_shape = { 1, 3,224,224 }; // input张量形状
    //Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape.data(), input_tensor_shape.size());

    //// 创建s_id张量
    //std::vector<int64_t> sid_tensor_shape = { 1 }; // s_id张量形状
    //std::vector<int32_t> sid_tensor_values = { 0 }; 
    //Ort::Value sid_tensor = Ort::Value::CreateTensor<int32_t>(memory_info, sid_tensor_values.data(), sid_tensor_values.size(), sid_tensor_shape.data(), sid_tensor_shape.size());

    //// 设置输入节点名称
    //std::vector<const char*> input_node_names = { "input","s_id" };

    //// 创建输入张量数组
    //std::vector<Ort::Value> input_tensors;
    //input_tensors.push_back(std::move(input_tensor));
    //input_tensors.push_back(std::move(sid_tensor));

    //// 设置输出节点名称
    //std::vector<const char*> output_node_names = { "logits" };

    //// 进行推理
    //auto output_tensors = session.Run(Ort::RunOptions{ nullptr },  // 运行选项 为空即可
    //                                input_node_names.data(),    // 输入节点名称
    //                                input_tensors.data(),       // 输入张量
    //                                input_tensors.size(),       // 输入张量数量
    //                                output_node_names.data(),   // 输出节点名称
    //                                output_node_names.size()	 // 输出张量数量
    //                                 );



    //


    //// 获取输出张量
    //float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    //for (int i = 0; i < 2; i++)
    //    printf(" %f\n", floatarr[i]);






    return 0;
}

