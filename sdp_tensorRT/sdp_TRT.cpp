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

    // ��ȡ�����ļ�
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

    // ��������ʱ & ��������
    std::unique_ptr<nvinfer1::IRuntime> runtime{ nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()) };
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    assert(mEngine.get() != nullptr);

    //int nbLayers = mEngine->getNbLayers();
    //std::cout << "Number of layers: " << nbLayers << std::endl;

    //std::cout << "name: " << mEngine->getName() << std::endl;

    for (int i = 0; i < 5; i++)
    {
    
    // ��ȡ����ʼʱ���
    auto start = std::chrono::high_resolution_clock::now();

    // ����ִ��������
    std::unique_ptr<nvinfer1::IExecutionContext> context(mEngine->createExecutionContext());


    // ��ȡ�����С
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

    // ��ȡ�����С ��������Ŀռ䶼Ҫ���� 
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


    // ׼������
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

    // �������� �� �豸
    auto input_image{ util::RGBMatReader("E:/ʦ�ִ���/�ѹ鵵���ݼ�/myDataset/mya1/train/fake/1100.bmp", input_dims) };
    //input_image.read();
    auto input_buffer = input_image.process();
    std::unique_ptr<int32_t> s_buffer = std::make_unique<int32_t>(0);

    // �������ݵ��豸
    cudaMemcpy(input_mem, input_buffer.get(), input_size, cudaMemcpyHostToDevice); // cudaMemcpyHostToDevice ���������豸 �� �ڴ浽�Դ�
    cudaMemcpy(s_mem, s_buffer.get(), s_size, cudaMemcpyHostToDevice);
    //cudaDeviceSynchronize(); // ����ͬ��


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


    // ����������ڴ� һ����������
    void* bindings[4];
    bindings[input_idx] = input_mem;
    bindings[s_idx] = s_mem;
    bindings[logits_idx] = logits_mem;
    bindings[pre_logits_idx] = pre_logits_mem;

    // ����
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

    // ��ȡ�������ʱ���
    auto end = std::chrono::high_resolution_clock::now();

    // �����������ʱ�䣬�Ժ���Ϊ��λ
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // �������ʱ��
    std::cout << "��������ʱ��Ϊ��" << duration.count() << " ����\n";



    // �ͷ�CUDA�ڴ�
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