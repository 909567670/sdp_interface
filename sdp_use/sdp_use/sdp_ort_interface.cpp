#include "sdp_ort_interface.h"

#include <iostream>
#include <opencv2/opencv.hpp>

sdp::logger sdp::sdp_ort_interface::logger;

sdp::sdp_ort_interface::sdp_ort_interface(const std::wstring& sdpPath, const std::wstring& vitPath):
	m_sdpPath{ sdpPath }, m_vitPath{vitPath}
{
	createEnv();
    if (createSessions(true))
	{
		//std::cout << "[SORTI]\tSdp OnnxRunTime Interface Init Success!" << std::endl;
		//std::cout << "[SORTI]\tRunning On CUDA!" << std::endl;
		logger.log("Sdp OnnxRunTime Interface Init Success!", logger::LOG_TYPE::LOG_INFO);
		logger.log("Running On CUDA!", logger::LOG_TYPE::LOG_INFO);
	}
	else
	{
		if (createSessions(false))
		{
			/*std::cout << "[SORTI]\tSdp OnnxRunTime Interface Init Success!" << std::endl;
			std::cout << "[SORTI]\tRunning On CPU!" << std::endl;*/
			logger.log("Sdp OnnxRunTime Interface Init Success!", logger::LOG_TYPE::LOG_INFO);
			logger.log("Running On CPU!", logger::LOG_TYPE::LOG_INFO);
		}
		else
		{
			//std::cout << "[SORTI]\t#### Sdp OnnxRunTime Interface Init Fail!" << std::endl;
			logger.log("Sdp OnnxRunTime Interface Init Fail!", logger::LOG_TYPE::LOG_ERROR);
		}
	}

	m_img_shape = { 1, 3,224,224 };
    m_sid_shape = { 1 };
}

sdp::sdp_ort_interface::sdp_ort_interface(sdp_ort_interface &&other) noexcept
    : m_pEnv(std::move(other.m_pEnv)),
    m_pVitSession(std::move(other.m_pVitSession)),
    m_pSdpSession(std::move(other.m_pSdpSession)),
    m_sdpPath(std::move(other.m_sdpPath)),
    m_vitPath(std::move(other.m_vitPath)),
    m_img_shape(std::move(other.m_img_shape)),
    m_sid_shape(std::move(other.m_sid_shape))
{
    // ÓÉÓÚloggerÊÇ¾²Ì¬µÄ£¬ËùÒÔ²»ÐèÒªÔÚÒÆ¶¯¹¹Ôìº¯ÊýÖÐ´¦Àí
    // ¶ÔÓÚÆäËû·Ç×ÊÔ´³ÉÔ±£¨ÈçuseCudaºÍuseTensorRT£©£¬Èç¹ûÓÐµÄ»°£¬Ò²Ó¦¸ÃÔÚÕâÀïÒÆ¶¯
}

sdp::sdp_ort_interface &sdp::sdp_ort_interface::operator=(sdp_ort_interface &&other) noexcept
{
    if (this != &other) {
        m_pEnv = std::move(other.m_pEnv);
        m_pVitSession = std::move(other.m_pVitSession);
        m_pSdpSession = std::move(other.m_pSdpSession);
        m_sdpPath = std::move(other.m_sdpPath);
        m_vitPath = std::move(other.m_vitPath);
        m_img_shape = std::move(other.m_img_shape);
        m_sid_shape = std::move(other.m_sid_shape);
        // ÓÉÓÚloggerÊÇ¾²Ì¬µÄ£¬ËùÒÔ²»ÐèÒªÔÚÒÆ¶¯¸³ÖµÔËËã·ûÖÐ´¦Àí
        // ¶ÔÓÚÆäËû·Ç×ÊÔ´³ÉÔ±£¨ÈçuseCudaºÍuseTensorRT£©£¬Èç¹ûÓÐµÄ»°£¬Ò²Ó¦¸ÃÔÚÕâÀïÒÆ¶¯
    }
    return *this;
}

bool sdp::sdp_ort_interface::vit_sidInfer(const cv::Mat& blob, int32_t& sid)
{
	// ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
	auto data_ptr = reinterpret_cast<float_t*>(blob.data);
	int data_size = blob.total() * blob.channels();
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float_t>(memory_info, data_ptr, data_size, m_img_shape.data(), m_img_shape.size());

	// ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
	std::vector<const char*> input_node_names = { "input" };
	std::vector<Ort::Value> input_tensors; 
	input_tensors.push_back(std::move(input_tensor)); 
	// ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
	std::vector<const char*> output_node_names = { "s_id" }; 
	// ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
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
		logger.log("sdpInfer: sid Error!", logger::LOG_TYPE::LOG_ERROR);
		return false;
	}

	// ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
	auto data_ptr = reinterpret_cast<float_t*>(blob.data);
	int data_size = blob.total() * blob.channels();
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float_t>(memory_info, data_ptr, data_size, m_img_shape.data(), m_img_shape.size());

	Ort::Value sid_tensor = Ort::Value::CreateTensor<int32_t>(memory_info, &sid, 1, m_sid_shape.data(), m_sid_shape.size());



	// ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
	std::vector<const char*> input_node_names = { "input" ,"s_id" };
	std::vector<Ort::Value> input_tensors; 
	input_tensors.push_back(std::move(input_tensor)); 
	input_tensors.push_back(std::move(sid_tensor));

	// ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
	std::vector<const char*> output_node_names = { "logits" };

	// ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
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
		logger.log("infer: Img is Empty or Format Error!", logger::LOG_TYPE::LOG_ERROR);
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
		logger.log("vit_sidInfer: Vit Infer Fail!", logger::LOG_TYPE::LOG_ERROR);
		return false;
	}

	//std::cout << "[SORTI]\tsid: " << sid << std::endl;
	logger.log("sid: " + std::to_string(sid), logger::LOG_TYPE::LOG_INFO);


	Ort::Value output_tensor(nullptr);
	if (!sdpInfer(blob, sid, output_tensor))
	{
		res = -1;
		//std::cout << "[SORTI]\t#### sdpInfer: sdp Infer Fail!" << std::endl;
		logger.log("sdpInfer: sdp Infer Fail!", logger::LOG_ERROR);
		return false;
	}

	auto output_tensor_ptr = output_tensor.GetTensorMutableData<float_t>();
	//std::cout << "[SORTI]\tlogits: " << "[ " << *output_tensor_ptr
	//	<< ", " << *(output_tensor_ptr + 1) << " ]" << std::endl;
	logger.log("logits: [ " + std::to_string(*output_tensor_ptr) + ", "
		+ std::to_string(*(output_tensor_ptr + 1)) + " ]", logger::LOG_TYPE::LOG_INFO);

	res = *output_tensor_ptr > *(output_tensor_ptr + 1) ? 0 : 1;

	return true;
}



cv::Mat sdp::sdp_ort_interface::imgProcess(cv::Mat& img)
{
	cv::Size imgSize(224, 224);
	cv::Mat blob; // ï¿½ï¿½ï¿½Ú´æ´¢Ô¤ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Í¼ï¿½ï¿½
	if (img.size() != imgSize)
	{
		cv::resize(img, blob, imgSize, 0, 0, cv::INTER_AREA); // Í¼ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ 
	}
	else
	{
		img.copyTo(blob);
	}
	//img.release(); 
	blob = cv::dnn::blobFromImage(blob, 1.0 / 255, blob.size(), cv::Scalar(0, 0, 0), true, false, CV_32F); // Í¼ï¿½ï¿½×ªÎªblobï¿½ï¿½Ê½ 

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
		//session_options.SetLogSeverityLevel(1); // ï¿½ï¿½Ö¾ï¿½ï¿½ï¿½ï¿½
		if (useCuda)
		{
			Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, device));
			//Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, device)); // ï¿½ï¿½ï¿½ï¿½
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
