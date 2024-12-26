#include "widget.h"
#include "sdp_interface.h"
#include <QCoreApplication>
#include <QDebug>
#include <QTextCodec>
#include <QTime>
#include "sdp_ort_interface.h"

// #define _T(str)  { QTextCodec::codecForName("GBK").fromUnicode(str).data() }

Widget::Widget(QWidget *parent)
    : QWidget(parent)
{
    //输出当前路径
    qDebug()<<QCoreApplication::applicationDirPath();
    sdp::SDPrompt_Interface sdp(
        "./models/vit.onnx",
        "./models/sdp_17.onnx",
        "./models/all_keys.npy"
        );
    // sdp.enableCuda();
    sdp.warmUp();

    sdp::sdp_ort_interface sdp_ort(
        L"./models/sdp_17.onnx",
        L"./models/vit_sid.onnx"
        );
    sdp_ort.warmUp();


    sdp::sdp_ort_interface sdp_ort2(
        L"./models/sdp_17.onnx",
        L"./models/vit_sid.onnx"
        );
    // sdp_ort2.warmUp();





    QString img_path = "E:/师兄代码/已归档数据集/myDataset/mya1/train/fake/1100.bmp";
    QTextCodec *codec = QTextCodec::codecForName("GBK");
    QByteArray encodedString = codec->fromUnicode(img_path);
    QString path = encodedString.data();

    cv::Mat img = cv::imread(_T("E:/师兄代码/已归档数据集/myDataset/mya1/train/fake/1100.bmp"));
    int result=-1;

    // 计算运行时间
    QTime time;
    time.start();
    sdp.getSDPromptResult_int(img,result);
    auto opencv_time = time.elapsed();

    // std::cout<<"opencv (cpu) time ="<<time.elapsed()<<"ms"<<std::endl;
    // qDebug()<<"res="<<result;




    int res = -1;

    time.restart();
    sdp_ort.infer(img, res);
    auto onnxrt_time = time.elapsed();



    sdp_ort2.infer(img, res);

    qDebug()<<"opencv (cpu) time ="<<opencv_time<<"ms";
    qDebug() << "onnxrt (gpu) time =" << onnxrt_time << "ms";



    // std::cout<< "onnxrt (gpu) time =" << time.elapsed() << "ms"<<std::endl;
    // qDebug() << "res=" << res;


}

Widget::~Widget() {}
