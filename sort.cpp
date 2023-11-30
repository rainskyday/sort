#include<iostream>
#include<opencv2/opencv.hpp>



using namespace std;
using namespace cv;


int main()
{
	std::vector<cv::Point> points = { cv::Point(100, 100), cv::Point(500, 100), cv::Point(500, 300),
								 cv::Point(500, 500), cv::Point(100, 500), cv::Point(200, 300) };
	std::vector<cv::Point> hull;
	cv::convexHull(points, hull);
	for (int i = 0; i < hull.size(); i++)
	{
		cout << hull[i] << endl;
	}

	cv::Mat image(600, 600, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::polylines(image, hull, true, cv::Scalar(0, 0, 255), 2);

	cv::imshow("Convex Hull", image);
	cv::waitKey(0);

	return 0;
}