
#include <iostream>

#include "sdp_interface.h"

#include<opencv2/opencv.hpp>
#include "cnpy.h"

int main()
{
	

	//cnpy::NpyArray arr = cnpy::npy_load("E:/S-DualPrompt/output/all_keys.npy"); 

	//float_t* data = arr.data<float_t>();
	//std::vector<int>shape;
	//for (const int& s : arr.shape)
	//{
	//	shape.push_back(s);
	//}
	//cv::Mat mat(3,&shape[0],CV_32F,data);



	//std::cout << "Hello World!\n";
	sdp::SDPrompt_Interface sdp_interface("E:/S-DualPrompt/output/vit.onnx",
									"E:/S-DualPrompt/output/sdp_17_ndcf.onnx",
									"E:/S-DualPrompt/output/all_keys.npy");
	/*cv::Mat f= cv::Mat::zeros(1, 768, CV_32F);
	cv::Mat s;
	sdp_interface.getSid(f,s); 
	std::cout << s << std::endl;*/


	//sdp_interface.enableCuda();
	sdp_interface.warmUp();

	cv::Mat img = cv::imread("E:/ʦ�ִ���/�ѹ鵵���ݼ�/myDataset/mya1/train/fake/1100.bmp");
	cv::Mat res;
	int result=-1;

	//
	//sdp_interface.getSdpResult(img, cv::Mat::zeros(1, 1, CV_8UC1), res); 
	//std::cout << res << std::endl;
	for (int i = 0; i < 10; i++)
	{
		// �����������ʱ��
		// ��ȡ����ʼʱ���
		auto start = std::chrono::high_resolution_clock::now();

		sdp_interface.getSDPromptResult_int(img, result);


		// ��ȡ�������ʱ���
		auto end = std::chrono::high_resolution_clock::now();

		// �����������ʱ�䣬�Ժ���Ϊ��λ
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

		// �������ʱ��
		std::cout << "��������ʱ��Ϊ��" << duration.count() << " ����\n";


		std::cout << result << std::endl;
	}
	return 0;
}
