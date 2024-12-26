#include "pch.h"
#include "framework.h"

#include "sdp_ort_interface.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>



sdp::sdp_ort_interface::sdp_ort_interface(const std::wstring& sdpPath, const std::wstring& vitPath):
	m_sdpPath{ sdpPath }, m_vitPath{vitPath}
{
	createEnv();
	if (createSessions(true))
	{
		//std::cout << "[SORTI]\tSdp OnnxRunTime Interface Init Success!" << std::endl;
		//std::cout << "[SORTI]\tRunning On CUDA!" << std::endl;
		m_logger.log("Sdp OnnxRunTime Interface Init Success!", logger::LOG_TYPE::LOG_INFO);
		m_logger.log("Running On CUDA!", logger::LOG_TYPE::LOG_INFO);
	}
	else
	{
		if (createSessions(false))
		{
			/*std::cout << "[SORTI]\tSdp OnnxRunTime Interface Init Success!" << std::endl;
			std::cout << "[SORTI]\tRunning On CPU!" << std::endl;*/
			m_logger.log("Sdp OnnxRunTime Interface Init Success!", logger::LOG_TYPE::LOG_INFO);
			m_logger.log("Running On CPU!", logger::LOG_TYPE::LOG_INFO);
		}
		else
		{
			//std::cout << "[SORTI]\t#### Sdp OnnxRunTime Interface Init Fail!" << std::endl;
			m_logger.log("Sdp OnnxRunTime Interface Init Fail!", logger::LOG_TYPE::LOG_ERROR);
		}
	}

	m_img_shape = { 1, 3,224,224 };
	m_sid_shape = { 1 };
}

bool sdp::sdp_ort_interface::vit_sidInfer(const cv::Mat& blob, int32_t& sid)
{
	// 输入数据
	auto data_ptr = reinterpret_cast<float_t*>(blob.data);
	int data_size = blob.total() * blob.channels();
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float_t>(memory_info, data_ptr, data_size, m_img_shape.data(), m_img_shape.size());

	// 设置输入
	std::vector<const char*> input_node_names = { "input" };
	std::vector<Ort::Value> input_tensors; 
	input_tensors.push_back(std::move(input_tensor)); 
	// 设置输出
	std::vector<const char*> output_node_names = { "s_id" }; 
	// 进行推理
	try {
		auto output_tensors = m_pVitSession->Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), input_tensors.size(), output_node_names.data(), output_node_names.size());
		auto output_tensor_ptr = output_tensors.front().GetTensorMutableData<int32_t>();
		sid = output_tensor_ptr[0];

	}
	catch (Ort::Exception e)
	{
		std::cerr << e.what() << std::endl;
		return false;
	}


	return true;
}

bool sdp::sdp_ort_interface::sdpInfer(const cv::Mat& blob, int32_t& sid, Ort::Value& resTensor)
{
	if (sid < 0)
	{
		//std::cout << "[SORTI]\t#### sdpInfer: sid Error!" << std::endl;
		m_logger.log("sdpInfer: sid Error!", logger::LOG_TYPE::LOG_ERROR);
		return false;
	}

	// 输入数据
	auto data_ptr = reinterpret_cast<float_t*>(blob.data);
	int data_size = blob.total() * blob.channels();
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float_t>(memory_info, data_ptr, data_size, m_img_shape.data(), m_img_shape.size());

	Ort::Value sid_tensor = Ort::Value::CreateTensor<int32_t>(memory_info, &sid, 1, m_sid_shape.data(), m_sid_shape.size());



	// 设置输入
	std::vector<const char*> input_node_names = { "input" ,"s_id" };
	std::vector<Ort::Value> input_tensors; 
	input_tensors.push_back(std::move(input_tensor)); 
	input_tensors.push_back(std::move(sid_tensor));

	// 设置输出
	std::vector<const char*> output_node_names = { "logits" };

	// 进行推理
	try {
		auto output_tensors = m_pSdpSession->Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), input_tensors.size(), output_node_names.data(), output_node_names.size()); 
		resTensor = std::move(output_tensors.front());
		
	}
	catch (Ort::Exception e) 
	{
		std::cerr << e.what() << std::endl; 
		return false;
	}


	
	return true;
}

bool sdp::sdp_ort_interface::infer(cv::Mat& img, int& res)
{
	if (img.empty() || img.type() != CV_8UC3)
	{
		res = -1;
		//std::cout << "[SORTI]\t#### infer: Img is Empty or Format Error!" << std::endl;
		m_logger.log("infer: Img is Empty or Format Error!", logger::LOG_TYPE::LOG_ERROR);
		return false;
	}

	cv::Mat blob = imgProcess(img);

	//float* data_ptr = reinterpret_cast<float*>(blob.data);
	//int num_elements = blob.total() * blob.channels();
	//for (int i = 0; i < num_elements; ++i) {
	//	 std::cout << data_ptr[i] << " "; 
	//}


	int sid = -1;
	if (!vit_sidInfer(blob, sid))
	{
		res = -1;
		//std::cout << "[SORTI]\t#### vit_sidInfer: Vit Infer Fail!" << std::endl;
		m_logger.log("vit_sidInfer: Vit Infer Fail!", logger::LOG_TYPE::LOG_ERROR);
		return false;
	}

	//std::cout << "[SORTI]\tsid: " << sid << std::endl;
	m_logger.log("sid: " + std::to_string(sid), logger::LOG_TYPE::LOG_INFO);


	Ort::Value output_tensor(nullptr);
	if (!sdpInfer(blob, sid, output_tensor))
	{
		res = -1;
		//std::cout << "[SORTI]\t#### sdpInfer: sdp Infer Fail!" << std::endl;
		m_logger.log("sdpInfer: sdp Infer Fail!", logger::LOG_ERROR);
		return false;
	}

	auto output_tensor_ptr = output_tensor.GetTensorMutableData<float_t>();
	//std::cout << "[SORTI]\tlogits: " << "[ " << *output_tensor_ptr
	//	<< ", " << *(output_tensor_ptr + 1) << " ]" << std::endl;
	m_logger.log("logits: [ " + std::to_string(*output_tensor_ptr) + ", " 
		+ std::to_string(*(output_tensor_ptr + 1)) + " ]", logger::LOG_TYPE::LOG_INFO);

	res = *output_tensor_ptr > *(output_tensor_ptr + 1) ? 0 : 1;

	return true;
}



cv::Mat sdp::sdp_ort_interface::imgProcess(cv::Mat& img)
{
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
	//img.release();
	blob = cv::dnn::blobFromImage(blob, 1.0 / 255, blob.size(), cv::Scalar(0, 0, 0), true, false, CV_32F); // 图像转为blob格式 

	return blob;
}

void sdp::sdp_ort_interface::warmUp()
{
	int res = -1;
	cv::Mat mat = cv::Mat::zeros(224, 224, CV_8UC3);
	vit_sidInfer(mat, res);
	auto temp = Ort::Value(nullptr);
	sdpInfer(mat, res, temp);
}

bool sdp::sdp_ort_interface::createSessions(bool useCuda = true, int device)
{
	try {
		Ort::SessionOptions session_options; 
		//session_options.SetLogSeverityLevel(1); // 日志级别
		if (useCuda)
		{
			Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, device));
		}
		m_pSdpSession = std::make_unique<Ort::Session>(*m_pEnv,m_sdpPath.c_str(),session_options);
		m_pVitSession = std::make_unique<Ort::Session>(*m_pEnv,m_vitPath.c_str(),session_options); 

	}
	catch (Ort::Exception e)
	{
		std::cerr << e.what() << std::endl;
		return false;
	}

	return true;
}
