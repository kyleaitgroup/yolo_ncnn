#include "yolo-fastestv2.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>  
#include <float.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <chrono>
int main()
{

    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    //在yolo-fastestv2.h 中, 有一个类函数  yoloFastestv2
    yoloFastestv2 api;
    //  读取模型
    api.loadModel("./model/yolo-fastestv2-opt.param",
                  "./model/yolo-fastestv2-opt.bin");
 
 
    // Vector容器中存放自定义数据类型,存放目标框的信息(cls,score,x,y,w,h,area), 在 yolo-fastestv2.h  有声明
    std::vector<TargetBox> boxes;
    cv::VideoCapture capture;

    capture.open(0);
    //获取当前 视频信息
    cv::Size S = cv::Size((int)capture.get(cv::CAP_PROP_FRAME_WIDTH),
                          (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT));
 
    // -----------保存视频的检测结果--------------
    // cv:: VideoWriter outputVideo;
    // outputVideo.open("./out.mp4", cv::VideoWriter::fourcc('P','I','M','1'), 30.0, S, true);
    // if (!outputVideo.isOpened()) {
    //     std::cout << "fail to open!" << std::endl;
    //     return -1;
    // }
    // ---------------------------------
 
    cv::Mat frame;



    double fps = 0.0;
    std::chrono::steady_clock::time_point start, end;
    int frameCount = 0;
    while (1) {

        start = std::chrono::steady_clock::now();
		capture >> frame;//读入视频的帧
		if (frame.empty()) break;
 
		// 检测 图像,结果保存在 boxes
        api.detection(frame, boxes);
        // 可视化,绘制框
        for (int i = 0; i < boxes.size(); i++) {
            std::cout<<boxes[i].x1<<" "<<boxes[i].y1<<" "<<boxes[i].x2<<" "<<boxes[i].y2
                     <<" "<<boxes[i].score<<" "<<boxes[i].cate<<std::endl;
 
            char text[256];
            sprintf(text, "%s %.1f%%", class_names[boxes[i].cate], boxes[i].score * 100);
 
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
 
            int x = boxes[i].x1;
            int y = boxes[i].y1 - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > frame.cols)
                x = frame.cols - label_size.width;
 
            cv::rectangle(frame, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(255, 255, 255), -1);
            cv::putText(frame, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
 
            cv::rectangle (frame, cv::Point(boxes[i].x1, boxes[i].y1),
                           cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255, 255, 0), 2, 2, 0);
        }



           // Calculate FPS
        end = std::chrono::steady_clock::now();
        double elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
        fps = 1.0 / elapsedTime;

        // Display FPS on the frame
        std::string fpsText = "FPS: " + std::to_string((int)fps);
        cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
 
        cv:: namedWindow("img",cv::WINDOW_NORMAL);
		cv:: imshow("img", frame);
 
         frameCount++;
 
		//按下ESC退出整个程序
  
        int c = cv::waitKey(30);
        if( char(c) == 27) return -1;
	}
//    cv::imwrite("output.png", cvImg);
    // 关闭释放
    capture.release();
    cv::waitKey(0);
    return 0;
}
 