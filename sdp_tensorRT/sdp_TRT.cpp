#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "util.h"
#include "logger.h"


constexpr long long operator"" _MiB(long long unsigned val)
{
    return val * (1 << 20);
}

using sample::gLogError;
using sample::gLogInfo;

using namespace nvinfer1;
//class Logger : public ILogger
//{
//    void log(Severity severity, const char* msg) noexcept override
//    {
//        // suppress info-level messages
//        if (severity <= Severity::kWARNING)
//            std::cout << msg << std::endl;
//    }
//};

int main()
{
    std::cout << "Hello, World!" << std::endl;

    // 读取引擎文件
    std::ifstream engineFile("sdp_17_ndcf.trt", std::ios::binary);
    if (engineFile.fail())
    {
        std::cerr << "Failed to open file!" << std::endl;
        return 0;
    }

    engineFile.seekg(0, std::ifstream::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    engineFile.close();

    // 创建运行时 & 加载引擎
    std::unique_ptr<nvinfer1::IRuntime> runtime{ nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()) };
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    assert(mEngine.get() != nullptr);

    //int nbLayers = mEngine->getNbLayers();
    //std::cout << "Number of layers: " << nbLayers << std::endl;

    //std::cout << "name: " << mEngine->getName() << std::endl;

    for (int i = 0; i < 5; i++)
    {
    
    // 获取程序开始时间点
    auto start = std::chrono::high_resolution_clock::now();

    // 创建执行上下文
    std::unique_ptr<nvinfer1::IExecutionContext> context(mEngine->createExecutionContext());


    // 获取输入大小
    auto input_idx = mEngine->getBindingIndex("input");
    if (input_idx == -1)
    {
        return false;
    }
    assert(mEngine->getBindingDataType(input_idx) == nvinfer1::DataType::kFLOAT);
    auto input_dims = context->getBindingDimensions(input_idx);
    context->setBindingDimensions(input_idx, input_dims);
    auto input_size = util::getMemorySize(input_dims, sizeof(float_t));

    auto s_idx = mEngine->getBindingIndex("s_id");
    if (s_idx == -1)
    {
        return false;
    }
    assert(mEngine->getBindingDataType(s_idx) == nvinfer1::DataType::kINT32);
    auto s_dims = context->getBindingDimensions(s_idx);

    context->setBindingDimensions(s_idx, s_dims);
    auto s_size = util::getMemorySize(s_dims, sizeof(int32_t));

    // 获取输出大小 所有输出的空间都要分配 
    auto logits_idx = mEngine->getBindingIndex("logits");
    if (logits_idx == -1)
    {
        return false;
    }
    assert(mEngine->getBindingDataType(logits_idx) == nvinfer1::DataType::kFLOAT);
    auto logits_dims = context->getBindingDimensions(logits_idx);
    auto logits_size = util::getMemorySize(logits_dims, sizeof(float_t));

    auto pre_logits_idx = mEngine->getBindingIndex("pre_logits");
    if (pre_logits_idx == -1)
    {
        return false;
    }
    assert(mEngine->getBindingDataType(pre_logits_idx) == nvinfer1::DataType::kFLOAT);
    auto pre_logits_dims = context->getBindingDimensions(pre_logits_idx);
    auto pre_logits_size = util::getMemorySize(pre_logits_dims, sizeof(float_t));


    // 准备推理
    // Allocate CUDA memory for input and output bindings
    void* input_mem{ nullptr };
    if (cudaMalloc(&input_mem, input_size) != cudaSuccess)
    {
        gLogError << "ERROR: input cuda memory allocation failed, size = " << input_size << " bytes" << std::endl;
        return false;
    }
    void* s_mem{ nullptr };
    if (cudaMalloc(&s_mem, s_size) != cudaSuccess)
    {
        gLogError << "ERROR: input cuda memory allocation failed, size = " << s_size << " bytes" << std::endl;
        return false;
    }
    void* logits_mem{ nullptr };
    if (cudaMalloc(&logits_mem, logits_size) != cudaSuccess)
    {
        gLogError << "ERROR: output cuda memory allocation failed, size = " << logits_size << " bytes" << std::endl;
        return false;
    }
    void* pre_logits_mem{ nullptr };
    if (cudaMalloc(&pre_logits_mem, pre_logits_size) != cudaSuccess)
    {
        gLogError << "ERROR: output cuda memory allocation failed, size = " << pre_logits_size << " bytes" << std::endl;
        return false;
    }

    // 输入数据 到 设备
    auto input_image{ util::RGBMatReader("E:/师兄代码/已归档数据集/myDataset/mya1/train/fake/1100.bmp", input_dims) };
    //input_image.read();
    auto input_buffer = input_image.process();
    std::unique_ptr<int32_t> s_buffer = std::make_unique<int32_t>(0);

    // 复制数据到设备
    cudaMemcpy(input_mem, input_buffer.get(), input_size, cudaMemcpyHostToDevice); // cudaMemcpyHostToDevice 从主机到设备 即 内存到显存
    cudaMemcpy(s_mem, s_buffer.get(), s_size, cudaMemcpyHostToDevice);
    //cudaDeviceSynchronize(); // 数据同步


    //auto input_buf = std::unique_ptr<float_t>{ new float_t[input_size / sizeof(float_t)] };
    //auto s_buf = std::unique_ptr<float_t>{ new float_t[s_size / sizeof(int32_t)] };
    //cudaMemcpyAsync(input_buf.get(), input_mem, input_size, cudaMemcpyDeviceToHost);
    //cudaMemcpyAsync(s_buf.get(), s_mem, s_size, cudaMemcpyDeviceToHost);
    //cudaDeviceSynchronize();

    //// Print the results
    //std::ofstream file("input.csv");
    //std::cout << "\input: " << input_size / sizeof(float_t) << std::endl;
    //for (size_t i = 0; i < input_size / sizeof(float_t); ++i) {
    //    //std::cout << input_buffer.get()[i] << std::endl;
    //    file << input_buf.get()[i] << std::endl;
    //}

    //std::cout << "\npre_logits:" << std::endl;
    //for (size_t i = 0; i < s_size / sizeof(int32_t); ++i) {
    //    std::cout << s_buffer.get()[i] << std::endl;
    //}
    //file.close();


    // 绑定输入输出内存 一起送入推理
    void* bindings[4];
    bindings[input_idx] = input_mem;
    bindings[s_idx] = s_mem;
    bindings[logits_idx] = logits_mem;
    bindings[pre_logits_idx] = pre_logits_mem;

    // 推理
    bool status = context->executeV2(bindings);

    if (!status)
    {
        gLogError << "ERROR: inference failed" << std::endl;
        cudaFree(input_mem);
        cudaFree(s_mem);
        cudaFree(logits_mem);
        cudaFree(pre_logits_mem);
        return 0;
    }

    auto logits_buffer = std::unique_ptr<float_t>{ new float_t[logits_size / sizeof(float_t)] };
    auto pre_logits_buffer = std::unique_ptr<float_t>{ new float_t[pre_logits_size / sizeof(float_t)] };
    cudaMemcpy(logits_buffer.get(), logits_mem, logits_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(pre_logits_buffer.get(), pre_logits_mem, pre_logits_size, cudaMemcpyDeviceToHost);
    //cudaDeviceSynchronize();

    // 获取程序结束时间点
    auto end = std::chrono::high_resolution_clock::now();

    // 计算程序运行时间，以毫秒为单位
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // 输出运行时间
    std::cout << "程序运行时间为：" << duration.count() << " 毫秒\n";



    // 释放CUDA内存
    cudaFree(input_mem);
    cudaFree(s_mem);
    cudaFree(logits_mem);
    cudaFree(pre_logits_mem);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        gLogError << "ERROR: failed to free CUDA memory: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Print the results
    std::cout << "\nlogits:" << std::endl;
    for (size_t i = 0; i < logits_size / sizeof(float_t); ++i) {
        std::cout << logits_buffer.get()[i] << std::endl;
    }

    //std::cout << "\npre_logits:" << std::endl;
    //for (size_t i = 0; i < pre_logits_size / sizeof(float_t); ++i) {
    //    std::cout << "[" <<i<<"]\t" << pre_logits_buffer.get()[i] << std::endl;
    //}
}
	return 0;
}