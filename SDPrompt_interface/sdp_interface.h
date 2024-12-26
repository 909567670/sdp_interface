
#ifndef SDP_INTERFACE_H
#define SDP_INTERFACE_H

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

	class SDPrompt_Interface
	{
	public:
		SDPrompt_Interface() {};
		SDPrompt_Interface(const std::string& vitPath,
			const std::string& sdpPath,
			const std::string& npyPath);
		~SDPrompt_Interface() {};

	private:
		sdp::Net m_vit; // 预训练vit
		sdp::Net m_sdp; // S-Dual Prompt
		bool m_cudaEnable = false; // 是否启用cuda

		cv::Mat m_all_keys; // 聚类中心

	public:
		// 启用cuda
		void enableCuda();

		bool load(const std::string& modelPath, sdp::Net& net);
		// 打印模型层信息

		void printNetLayers(sdp::Net& net);

		// vit 获取特征
		bool getVitFeature(const cv::Mat& img, cv::Mat& feature);

		// sdp 获取结果 
		bool getSdpResult(const cv::Mat& img, const cv::Mat& sid, cv::Mat& result);

		// 获取 s_id
		void getSid(const cv::Mat& feature, cv::Mat& sid);

		// 获得异物检测结果
		bool getSDPromptResult_mat(const cv::Mat& img, cv::Mat& result);
		bool getSDPromptResult_int(const cv::Mat& img, int& result);


		// 预热模型
		void warmUp();

		// 读取聚类中心 npy
		bool loadNpy(const std::string& path, cv::Mat& mat);
	};
}








#endif // SDP_INTERFACE_H