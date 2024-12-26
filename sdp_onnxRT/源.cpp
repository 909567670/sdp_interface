//#include <onnxruntime/core/providers/cuda/cuda_provider_factory.h>
//#include <onnxruntime_cxx_api.h>

#include <iostream>
#include "opencv2/opencv.hpp"

#include "sdp_ort_interface.h"

int main()
{

    sdp::sdp_ort_interface sdp(L"sdp_17.onnx", L"vit_sid.onnx");
    int res;
    cv::Mat mat = cv::imread("E:/ʦ�ִ���/�ѹ鵵���ݼ�/myDataset/mya1/train/fake/1100.bmp");
    //sdp.warmUp();
    // �����������ʱ��
   // ��ȡ����ʼʱ���

    for (int i = 0; i < 100; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        sdp.infer(mat, res);

        // ��ȡ�������ʱ���
        auto end = std::chrono::high_resolution_clock::now();

        // �����������ʱ�䣬�Ժ���Ϊ��λ
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // �������ʱ��
        std::cout << "��������ʱ��Ϊ��" << duration.count() << " ����\n";
        std::cout << "res: " << res << std::endl;

        Sleep(500);
    }

    //// ��ʼ��ONNX Runtime
    //Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sdp infer");

    //// �����Ựѡ����CUDA�ṩ����
    //Ort::SessionOptions session_options;
    //uint32_t device_id = 0; // CUDA �豸 ID
    ////Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, device_id));
    //Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, device_id));

    //// ����ģ��
    //const std::wstring model_path = L"sdp_17.onnx";
    //Ort::Session session(env, model_path.c_str(), session_options);
    //    



   
    //// ���ڱ���input����ֵ
    //std::vector<float> input_tensor_values;

    //// ������������
    //cv::Mat img = cv::imread("E:/ʦ�ִ���/�ѹ鵵���ݼ�/myDataset/mya1/train/fake/1100.bmp");
    //cv::Size imgSize(224, 224); 
    //cv::Mat blob; // ���ڴ洢Ԥ������ͼ��
    //if (img.size() != imgSize) 
    //{
    //    cv::resize(img, blob, imgSize, 0, 0, cv::INTER_AREA); // ͼ������ 
    //}
    //else
    //{
    //    img.copyTo(blob); 
    //}

    //blob = cv::dnn::blobFromImage(blob, 1.0 / 255, blob.size(), cv::Scalar(0, 0, 0), true, false, CV_32F); // ͼ��תΪblob��ʽ 

    //float* data_ptr = reinterpret_cast<float*>(blob.data); // ��ȡblob����ָ��
    //int num_elements = blob.total() * blob.channels(); // ��ȡblob����Ԫ�ظ���
    //// ��blob���ݿ�����input������
    //for (int i = 0; i < num_elements; ++i) 
    //{
    //    /*std::cout << data_ptr[i] << " ";*/
    //    input_tensor_values.push_back(std::move(data_ptr[i]));
    //}

    //// �����ڴ���Ϣ
    //auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    //
    //// ����input����
    //std::vector<int64_t> input_tensor_shape = { 1, 3,224,224 }; // input������״
    //Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape.data(), input_tensor_shape.size());

    //// ����s_id����
    //std::vector<int64_t> sid_tensor_shape = { 1 }; // s_id������״
    //std::vector<int32_t> sid_tensor_values = { 0 }; 
    //Ort::Value sid_tensor = Ort::Value::CreateTensor<int32_t>(memory_info, sid_tensor_values.data(), sid_tensor_values.size(), sid_tensor_shape.data(), sid_tensor_shape.size());

    //// ��������ڵ�����
    //std::vector<const char*> input_node_names = { "input","s_id" };

    //// ����������������
    //std::vector<Ort::Value> input_tensors;
    //input_tensors.push_back(std::move(input_tensor));
    //input_tensors.push_back(std::move(sid_tensor));

    //// ��������ڵ�����
    //std::vector<const char*> output_node_names = { "logits" };

    //// ��������
    //auto output_tensors = session.Run(Ort::RunOptions{ nullptr },  // ����ѡ�� Ϊ�ռ���
    //                                input_node_names.data(),    // ����ڵ�����
    //                                input_tensors.data(),       // ��������
    //                                input_tensors.size(),       // ������������
    //                                output_node_names.data(),   // ����ڵ�����
    //                                output_node_names.size()	 // �����������
    //                                 );



    //


    //// ��ȡ�������
    //float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    //for (int i = 0; i < 2; i++)
    //    printf(" %f\n", floatarr[i]);






    return 0;
}

