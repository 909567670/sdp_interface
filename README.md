# sdp_interface
sdp模型部署C++工程, 实现了OpenCV / ORT / TensorRT库
vit和sdp模型

vit: img -> sid

sdp: img,sid -> logits 

## ORT 
完整实现

## OpenCV (旧)
vit输出图像特征, 需配合聚类中心npy文件, 计算sid

## TRT
只实现sdp, 输出logits异常

## 库版本
- OpenCV: 4.9.0 (cuda)
- TRT: 8.6.1.6
- ORT: 1.17.1_cuda12
