#ifndef __SDP_ORT_INTERFACE_H__ 
#define __SDP_ORT_INTERFACE_H__

#include <core/session/onnxruntime_cxx_api.h>
#include <windows.h>
#include <iostream>
#include <mutex>

// ǰ������
namespace cv {
	class Mat;
};

namespace sdp {

	class logger
	{
	private:
		std::mutex logMutex;

	public:
		enum LOG_TYPE
		{
			LOG_ERROR,
			LOG_INFO,
		};

		inline void log(std::string s, LOG_TYPE type)
		{
			std::lock_guard<std::mutex> lock(logMutex);
			HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
			// ��ȡ��ǰʱ��
			SYSTEMTIME sys;
			GetLocalTime(&sys);
			std::string time = std::to_string(sys.wYear) + "-" + std::to_string(sys.wMonth) + "-" + std::to_string(sys.wDay) + " "
				+ std::to_string(sys.wHour) + ":" + std::to_string(sys.wMinute) + ":" + std::to_string(sys.wSecond)+"."+std::to_string(sys.wMilliseconds);

			// �����������Log
			switch (type)
			{
			case LOG_ERROR:
				SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_INTENSITY); // Red
				std::cerr << time << " [SORTI] [E] " << s << std::endl;
				break;
			case LOG_INFO:
				SetConsoleTextAttribute(hConsole, FOREGROUND_GREEN | FOREGROUND_INTENSITY); // Green
				std::cout << time << " [SORTI] [I] " << s << std::endl;
				break;
			default:
				break;
			}

			// Reset to white
			SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
		}
	};




	class sdp_ort_interface
	{
	public:
		sdp_ort_interface(const std::wstring& sdpPath,
			const std::wstring& vitPath);

		// vit_sid ģ������ 
		// ���� (Mat)img 
		// ��� (int32_t)sid
		bool vit_sidInfer(const cv::Mat& blob, int32_t& sid);
		// sdp ģ������ 
		// ���� (Mat)img ��(int32_t)sid 
		// ��� (float_t*)resTensor
		bool sdpInfer(const cv::Mat& blob, int32_t& sid, Ort::Value& resTensor);

		// ͼ��Ԥ����
		cv::Mat imgProcess(cv::Mat& img);


		// ģ������
		// ���� (Mat)img 
		// ��� (int)res 0 ������ 1 ������
		bool infer(cv::Mat& img, int& res);

		// Ԥ�� 
		void warmUp();
	private:
		std::unique_ptr<Ort::Env> m_pEnv; // ort ���� Ψһ
		std::unique_ptr<Ort::Session> m_pVitSession;
		std::unique_ptr<Ort::Session> m_pSdpSession;

		std::wstring m_sdpPath;
		std::wstring m_vitPath;

		std::vector<int64_t> m_img_shape = { 1, 3,224,224 };
		std::vector<int64_t> m_sid_shape = { 1 };

		static sdp::logger logger;

		// ����
		/*bool useCuda = false;
		bool useTensorRT = false;*/

	private:
		// ���� ort ����
		inline void createEnv() { m_pEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "sdp ort infer"); };
		// ���� ort �Ự(vit & sdp) 
		bool createSessions(bool useCuda, int device = 0);
	};




};


#endif