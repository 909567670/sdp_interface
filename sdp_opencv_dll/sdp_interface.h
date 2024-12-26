


#ifndef SDP_INTERFACE_H
#define SDP_INTERFACE_H

//#ifdef __cplusplus //����һ���־��Ǹ��߱����������������__cplusplus(�������cpp�ļ��� 
//extern "C" { //��Ϊcpp�ļ�Ĭ�϶����˸ú�),�����C���Է�ʽ���б���
//#endif

#ifdef SDPOPENCVDLL_EXPORTS
#define SDPOPENCVDLL_API __declspec(dllexport)
#else
#define SDPOPENCVDLL_API __declspec(dllimport)
#endif


#include <opencv2/opencv.hpp>
#include "cnpy.h"
namespace sdp
{
	struct Net
	{
		cv::dnn::Net net;
		std::string path;
		//bool useCuda = false;
	};

	class SDPOPENCVDLL_API SDPrompt_Interface
	{
	public:
		SDPrompt_Interface() {};
		SDPrompt_Interface(const std::string& vitPath,
			const std::string& sdpPath,
			const std::string& npyPath);
		~SDPrompt_Interface() {};

	private:
		sdp::Net m_vit; // Ԥѵ��vit
		sdp::Net m_sdp; // S-Dual Prompt
		bool m_cudaEnable = false; // �Ƿ�����cuda

		cv::Mat m_all_keys; // ��������

	public:
		// ����cuda
		void enableCuda();

		bool load(const std::string& modelPath, sdp::Net& net);
		// ��ӡģ�Ͳ���Ϣ

		void printNetLayers(sdp::Net& net);

		// vit ��ȡ����
		bool getVitFeature(const cv::Mat& img, cv::Mat& feature);

		// sdp ��ȡ��� 
		bool getSdpResult(const cv::Mat& img, const cv::Mat& sid, cv::Mat& result);

		// ��ȡ s_id
		void getSid(const cv::Mat& feature, cv::Mat& sid);

		// �����������
		bool getSDPromptResult_mat(const cv::Mat& img, cv::Mat& result);
		bool getSDPromptResult_int(const cv::Mat& img, int& result);


		// Ԥ��ģ��
		void warmUp();

		// ��ȡ�������� npy
		bool loadNpy(const std::string& path, cv::Mat& mat);
	};
}








#endif // SDP_INTERFACE_H