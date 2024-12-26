#ifndef __SDP_ORT_INTERFACE_H__ 
#define __SDP_ORT_INTERFACE_H__

#ifdef SDPONNXRTDLL_EXPORTS
#define SDPONNXRTDLL_API __declspec(dllexport)
#else
#define SDPONNXRTDLL_API __declspec(dllimport)
#endif


#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <windows.h>
#include <iostream>

//// 前置声明
//namespace cv {
//	class Mat;
//};
//namespace Ort {
//	class Env;
//	class Session;
//};

namespace sdp {

	class logger
	{
	public:
		enum LOG_TYPE
		{
			LOG_ERROR,
			LOG_INFO,
		};

		inline void log(std::string s, LOG_TYPE type)
		{
			HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
			// 获取当前时间
			SYSTEMTIME sys;
			GetLocalTime(&sys);
			std::string time = std::to_string(sys.wYear) + "-" + std::to_string(sys.wMonth) + "-" + std::to_string(sys.wDay) + " "
				+ std::to_string(sys.wHour) + ":" + std::to_string(sys.wMinute) + ":" + std::to_string(sys.wSecond)+"."+std::to_string(sys.wMilliseconds);

			switch (type)
			{
			case LOG_ERROR:
				SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_INTENSITY);
				std::cerr << time << " [SORTI] [E] " << s << std::endl;
				break;
			case LOG_INFO:
				SetConsoleTextAttribute(hConsole, FOREGROUND_GREEN | FOREGROUND_INTENSITY);
				std::cout << time << " [SORTI] [I] " << s << std::endl;
				break;
			default:
				break;
			}

			// Reset to white
			SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
		}
	};




	class  sdp_ort_interface
	{
	public:
		SDPONNXRTDLL_API sdp_ort_interface(const std::wstring& sdpPath,
			const std::wstring& vitPath);

		// vit_sid 模型推理 
		// 输入 (Mat)img 
		// 输出 (int32_t)sid
		SDPONNXRTDLL_API bool vit_sidInfer(const cv::Mat& blob, int32_t& sid);
		// sdp 模型推理 
		// 输入 (Mat)img 、(int32_t)sid 
		// 输出 (float_t*)resTensor
		SDPONNXRTDLL_API bool sdpInfer(const cv::Mat& blob, int32_t& sid, Ort::Value& resTensor);

		// 图像预处理
		SDPONNXRTDLL_API cv::Mat imgProcess(cv::Mat& img);


		// 模型推理
		// 输入 (Mat)img 
		// 输出 (int)res 0 无异物 1 有异物
		SDPONNXRTDLL_API bool infer(cv::Mat& img, int& res);

		// 预热 
		SDPONNXRTDLL_API void warmUp();
	private:
		std::unique_ptr<Ort::Env> m_pEnv; // ort 环境 唯一
		std::unique_ptr<Ort::Session> m_pVitSession;
		std::unique_ptr<Ort::Session> m_pSdpSession;

		std::wstring m_sdpPath;
		std::wstring m_vitPath;

		std::vector<int64_t> m_img_shape = { 1, 3,224,224 };
		std::vector<int64_t> m_sid_shape = { 1 };

		sdp::logger m_logger;

	private:
		// 创建 ort 环境
		inline void createEnv() { m_pEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "sdp ort infer"); };
		// 创建 ort 会话(vit & sdp) 
		bool createSessions(bool useCuda, int device = 0);
	};




};


#endif