#include<iostream>
#include<opencv2/opencv.hpp>



using namespace std;
using namespace cv;

//匹配度计算函数，欧几里得距离
float calculateEuclideanDistance(const std::vector<float>& vector1, const std::vector<float>& vector2) 
{
	if (vector1.size() != vector2.size()) {
		std::cerr << "Error: Vectors must have the same size." << std::endl;
		return -1.0f;
	}

	float squaredSum = 0.0f;
	for (size_t i = 0; i < vector1.size(); i++) {
		float diff = vector1[i] - vector2[i];
		squaredSum += diff * diff;
	}

	return std::sqrt(squaredSum);
}
//计算相对于最小外切圆圆心作为极坐标原点的角度
double calculateAngle(cv::Point center, cv::Point point) {
    double dx = point.x - center.x;
    double dy = point.y - center.y;
    return atan2(dy, dx) * 180 / CV_PI;
}
//计算绘制出的封闭图形的角度
double calculateAngle_2(cv::Point point1, cv::Point point2, cv::Point point3) {
    double dx1 = point1.x - point2.x;
    double dy1 = point1.y - point2.y;
    double dx2 = point3.x - point2.x;
    double dy2 = point3.y - point2.y;

    double dotProduct = dx1 * dx2 + dy1 * dy2;
    double magnitude1 = sqrt(dx1 * dx1 + dy1 * dy1);
    double magnitude2 = sqrt(dx2 * dx2 + dy2 * dy2);

    double cosTheta = dotProduct / (magnitude1 * magnitude2);
    return acos(cosTheta) * 180 / CV_PI;
}
//计算两点欧氏距离
double getdis(Point2f p1, Point2f p2)
{
	double dis = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
	return dis;
}
//按某点旋转
Point2f rotation(Point2f p, double theta)
{
	Point2f p_rot;
	p_rot.x = p.x * cos(theta) - p.y * sin(theta);
	p_rot.y = p.x * sin(theta) + p.y * cos(theta);
	return p_rot;
}
//求距离矩阵
vector<vector<double>> compute_dis_mat(vector<Point2f> points)
{
	vector<vector<double>> dis_mat;
	for (int i = 0; i < points.size(); i++)
	{
		for (int j = 0; j < points.size(); j++)
		{
			if (i == j)
			{
				dis_mat[i][j] = 0;
			}
			else
			{
				double tmp = sqrt(pow(points[i].x - points[j].x, 2) + pow(points[i].y - points[j].y, 2));
				dis_mat[i][j] = tmp;
			}
		}

	}

	return dis_mat;
}
//结构体定义
struct SegmentPoint
{
	vector<Point2f> imgpoints;
	Mat img_seg;
	int t;
	int tf;
};
//图像分割及圆心定位：主灯阵
SegmentPoint Segmentation(Mat img, float k, int b, int min_area)
{

	SegmentPoint return_data;
	//float k = 0.6f;
	//int b = 0;
	//int min_area = 10;
	//Mat img = imread("D:\\Jupyter\\LOS_rongyu\\fuke\\LOS_rongyu\\2.png", 1);
	//Mat img;
	Mat img2 = img.clone();
	Mat grayimg;//灰度图
	Mat thresh;//
	cvtColor(img2, grayimg, COLOR_BGR2GRAY);
	double minvalue = 0.0;
	double maxvalue = 0.0;
	int minLoc[2], maxLoc[2];
	minMaxIdx(grayimg, &minvalue, &maxvalue, minLoc, maxLoc);//找最大灰度
	int gray_th = maxvalue * k;//灰度阈值
	//cout << gray_th << endl;
	//大津法，用设置的灰度阈值将图像二值化处理
	threshold(grayimg, thresh, gray_th, 255, THRESH_BINARY);
	//imshow("二值化", thresh);
	//waitKey(0);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat openimg, sure_bg;//
	morphologyEx(thresh, openimg, MORPH_OPEN, kernel);
	//
	dilate(openimg, sure_bg, kernel);
	//imshow("sure_bg", sure_bg);
	//waitKey(0);
	Mat dist_transform, sure_fg;
	distanceTransform(openimg, dist_transform, DIST_L2, 5);//计算距零像素点的距离，通过设定合理的阈值对距离变换后的图像进行二值化处理
	threshold(dist_transform, sure_fg, 0, 255, 0);//疑问1
	//imshow("dist_transform", dist_transform);
	//waitKey(0);
	Mat unknow;
	subtract(sure_bg, sure_fg, unknow, noArray(), CV_8UC1);
	//imshow("sure_fg", sure_fg);
	//imshow("unkonow", unknow);
	//waitKey(0);

	//分水岭算法

	//Mat markers;
	//sure_fg.convertTo(sure_bg, CV_8UC1);
	//int number = connectedComponents(sure_fg, markers, 8, CV_32S);
	//cout << number << endl;
	//Mat segment;
	//markers = markers + 1;	

	//watershed(img2, markers);
	//convertScaleAbs(markers,segment);
	//imshow("After Watershed", segment);
	Mat labels;
	int number = connectedComponents(unknow, labels, 8, CV_32S);


	//原图显示割线
	img2.setTo(Scalar(0, 0, 255), unknow == 255);
	//img2.setTo(Scalar(0, 0, 255), markers == -1);
	//imshow("color", img2);
	//cv::waitKey(0);

	//二值分割图
	Mat img_seg = sure_bg.clone();

	//img_seg.setTo(Scalar(0), sure_bg == 0);
	//img_seg.setTo(Scalar(255), sure_bg == 255);

	medianBlur(img_seg, img_seg, 5);
	//imshow("medianBlur", img_seg);

	//提取中心点

	vector<Point2f> center;
	vector<Mat> center_mask;
	vector<float> r;
	Mat stats, centroids;
	int num_stats = connectedComponentsWithStats(img_seg, labels, stats, centroids, 8);
	//cout << "Number of objects: " << num_stats - 1 << endl;
	for (int i = 0; i < num_stats; i++)
	{
		Mat mask = labels == i;
		if (countNonZero(img_seg & mask) == 0)
		{
			continue;
		}
		Mat img_zero = Mat::zeros(img2.size(), CV_8UC1);
		//imshow("mask", mask);
		//waitKey(0);
		img_zero.setTo(255, mask == 255);
		//imshow("img_zero", img_zero);
		//waitKey(0);
		vector<vector<Point>> contours;
		findContours(img_zero, contours, RETR_TREE, CHAIN_APPROX_NONE);
		double area = contourArea(contours[0]);
		Point2f center_point;
		float radius = 0.0f;
		minEnclosingCircle(contours[0], center_point, radius);
		double circle = area / (3.1415926 * radius * radius);
		//cout << circle << endl;
		int min_area = 10;

		if (min_area < area && area < img.size().area() && circle > 0.5)
		{
			center.push_back(center_point);
			center_mask.push_back(mask);
			r.push_back(radius);
			cv::circle(img2, center_point, 2, Scalar(100, 100, 0), -1);
		}
		else
		{
			img_seg &= ~mask;
		}

	}
	//cout << size(center) << endl;

	int t = 0;
	int tf = 0;
	if (center.size() >= 5)
	{
		RotatedRect box = minAreaRect(center);
		double length = max(box.size.width, box.size.height);
		double width = min(box.size.width, box.size.height);
		//cout << length << endl;
		//cout << width << endl;
		if (length / width > 3)//两个灯阵
		{
			int t = 1;
			Point2f vertexs[4];
			box.points(vertexs);
			width = 10000;
			int i_width = 0;
			vector<Point2f> new_center;
			//短边另一个点的索引
			//找到这个索引
			for (int i = 0; i < 3; i++)
			{
				double w = getdis(vertexs[0], vertexs[i + 1]);
				if (w < width)
				{
					i_width = i + 1;
					width = w;
				}
				//cout << w << endl;
			}
			for (int i = 0; i < center.size(); i++)
			{
				new_center.push_back(rotation(center[i], atan2(vertexs[0].y - vertexs[i_width].y, vertexs[0].x - vertexs[i_width].x)));
				//cout << new_center[i] << endl;
			}
			int delta_y = 0;
			int i_delete = center.size();
			//cout << "i_delete:" << i_delete << "new_center.size():" << new_center.size() << endl;
			for (int i = 0; i < new_center.size() - 1; i++)
			{
				double dy = 0 - (new_center[i + 1].y - new_center[i].y);
				//cout << "dy:" << dy << endl;
				if (dy > delta_y)
				{

					delta_y = dy;
					i_delete = i + 1;

				}
				//cout << "i_delete:" << i_delete << endl;
			}
			//cout << "center.size()" << center.size() << endl;


			if (center.size() - i_delete == 5)
			{
				tf = 1;

			}
			else { tf = 0; }
			//cout << "i_delete:" << i_delete << endl;
			//cout << "tf" << tf << endl;
			//cout << "center.size()" << center.size() << endl;
			center.erase(center.begin(), center.end() - (center.size() - i_delete));
			//cout << "center.size()" << center.size() << endl;
			//for (int i = 0; i < center.size(); i++)
			//{
			//	cout << center[i] << endl;
			//}

			//cout << "new_center.size()" << new_center.size() << endl;
			for (int i = 0; i < new_center.size() - i_delete; i++)
			{
				img_seg &= ~center_mask[i + i_delete];//疑问2这段写的有问题
			}
		}
	}


	//imshow("dstb", img_seg);

	vector<Point2f> imgpoint;
	vector<double> bb;
	//按距离保持灯阵数量小于等于6,需要的输入：中心点坐标，最小外接圆半径和对应的联通域图像
	if (center.size() <= 6)
	{
		imgpoint = center;
		//cout << imgpoint << endl;

	}
	else
	{
		vector<vector<double>> Distance_mat = compute_dis_mat(center);
		//cout << Distance_mat << endl;
		for (int i = 0; i < Distance_mat.size(); i++)
		{
			for (int j = 0; j < Distance_mat[i].size(); j++)
			{
				bb[i] += Distance_mat[i][j];
			}
		}

		//求完bb了


		//用c排序,下标
		vector<int> c(center.size());
		for (int i = 0; i < center.size(); i++)
		{
			int index = -1;
			double max_distance = 0;
			for (int j = 0; j < center.size(); j++)
			{
				if (bb[j] > max_distance)
				{
					max_distance = bb[j];
					index = j;
				}
			}
			bb[index] = 0;
			c[i] = index;
		}//现在c是按从大到小的角标排列 最大的是0

		vector<Point2f> d(center.size());
		for (int i = 0; i < center.size() - 6; i++)
		{
			d[i] = center[c[i]];
		}//d里是要删除的部分,现在是d里和center里都有

		reverse(c.begin(), c.end());//倒序
		for (int i = 0; i < 6; i++)
		{
			imgpoint[i] = center[c[i]];
		}
		//center.swap(d);//互换两个vector		
		//for (int i = 0; i < center.size(); i++)
		//{

		//	imgpoint[i] =  center[c[i]];
		//}


	}




	return_data.imgpoints = imgpoint;
	return_data.img_seg = img_seg;
	return_data.t = t;
	return_data.tf = tf;

	return return_data;
	//waitKey(0);

}
//6灯排序
vector<Point2f> rank6light_main(Mat input_img, vector<Point2f> input_point)
{
	vector<Point2f> points = input_point;
	// 计算最小包络圆
	Point2f center;
	float radius;
	minEnclosingCircle(points, center, radius);
	// 创建一个绘制图形的图像
	Mat image = input_img.clone();
	// 绘制每个点和最小包络圆
	//for (int i = 0; i < points.size(); i++) 
	//{
	//    circle(image, points[i], 3, Scalar(0, 0, 255), -1);
	//}
	circle(image, center, radius, Scalar(0, 0, 255), 2);

	//vector<float> angles_zero;
	// 输出每个点相对于包络圆圆心的角度信息
	//for (int i = 0; i < points.size(); i++) {
	//    double angle = calculateAngle(center, points[i]);
	//    angles_zero.push_back(angle);
	//    std::cout << "Point " << i + 1 << ": " << angle << " degrees" << std::endl;
	//}

	// 按照角度信息从小到大排序点的向量
	sort(points.begin(), points.end(), [&](const cv::Point& a, const cv::Point& b) { return calculateAngle(center, a) < calculateAngle(center, b); });
	// 输出每个点相对于包络圆圆心的角度信息
	//for (int i = 0; i < points.size(); i++) {
	//    double angle = calculateAngle(center, points[i]);
	//    cout << "Point " << i + 1 << ": " << angle << " degrees" << endl;
	//}

	//绘制多边形
	vector<cv::Point> contour;
	for (int i = 0; i < points.size(); i++) {
		circle(image, points[i], 3, Scalar(0, 0, 255), -1);
		contour.push_back(points[i]);
	}
	cv::polylines(image, contour, true, cv::Scalar(0, 255, 0), 2);


	vector<float> real_angles;
	// 计算每个角并输出
	for (int i = 0; i < contour.size(); i++) {
		int prevIndex = (i == 0) ? contour.size() - 1 : i - 1;
		int nextIndex = (i == contour.size() - 1) ? 0 : i + 1;
		double angle = calculateAngle_2(contour[prevIndex], contour[i], contour[nextIndex]);
		real_angles.push_back(angle);
	}

	//输出多边形的每个角度值
	//for (int i = 0; i < real_angles.size(); i++) 
	//{
	//	cout << "Angle " << i + 1 << ": " << real_angles[i] << " degrees" << endl;
	//}
	//6灯阵仅存在的6种情况//其实应该能只写一个vector然后循环即可
	vector<vector<float>> model_6light = {
	{ 70,90,180,90,70,140 },//5 4 2 1 3 6
	{ 90,180,90,70,140,70 },//4 3 1 6 2 5 
	{ 180,90,70,140,70,90 },//3 2 6 5 1 4
	{ 90,70,140,70,90,180 },//2 1 5 4 6 3
	{ 70,140,70,90,180,90 },//1 6 4 3 5 2
	{ 140,70,90,180,90,70 } //6 5 3 2 4 1
	};
	vector<float> model1 = { 70,90,180,90,70,140 };
	//vector<float> model2 = { 90,180,90,70,140,70 };
	//vector<float> model3 = { 180,90,70,140,70,90 };
	//vector<float> model4 = { 90,70,140,70,90,180 };
	//vector<float> model5 = { 70,140,70,90,180,90 };
	//vector<float> model6 = { 140,70,90,180,90,70 };
	//cout << model_6light.size() << endl;
	vector<float> model_vector = model1;
	vector<float> models;//模型匹配临时变量
	vector<float> deltas;//欧几里得距离：恒定匹配程度，值越小则越接近
	//计算与各种情况的匹配程度
	for (int i = 0; i < model_6light.size(); i++)
	{
		models = model_vector;
		float delta = calculateEuclideanDistance(models, real_angles);
		deltas.push_back(delta);
		rotate(model_vector.rbegin(), model_vector.rbegin() + 1, model_vector.rend());
	}
	//求得匹配程度最高的情况并将其视为真实情况
	auto minElement = min_element(deltas.begin(), deltas.end());
	int minIndex = distance(deltas.begin(), minElement);
	vector<cv::Point2f> new_points;//按不同情况标注信息
	switch (minIndex)
	{
	case 0: {new_points.push_back(points[4]); new_points.push_back(points[3]); new_points.push_back(points[1]); new_points.push_back(points[0]); new_points.push_back(points[2]); new_points.push_back(points[5]); }; break;
	case 1: {new_points.push_back(points[5]); new_points.push_back(points[4]); new_points.push_back(points[2]); new_points.push_back(points[1]); new_points.push_back(points[3]); new_points.push_back(points[0]); }; break;
	case 2: {new_points.push_back(points[0]); new_points.push_back(points[5]); new_points.push_back(points[3]); new_points.push_back(points[2]); new_points.push_back(points[4]); new_points.push_back(points[1]); }; break;
	case 3: {new_points.push_back(points[1]); new_points.push_back(points[0]); new_points.push_back(points[4]); new_points.push_back(points[3]); new_points.push_back(points[5]); new_points.push_back(points[2]); }; break;
	case 4: {new_points.push_back(points[2]); new_points.push_back(points[1]); new_points.push_back(points[5]); new_points.push_back(points[4]); new_points.push_back(points[0]); new_points.push_back(points[3]); }; break;
	case 5: {new_points.push_back(points[3]); new_points.push_back(points[2]); new_points.push_back(points[0]); new_points.push_back(points[5]); new_points.push_back(points[1]); new_points.push_back(points[4]); }; break;
	}

	//输出检测
	for (int i = 0; i < new_points.size(); i++)
	{
		cout << "Angle " << i + 1 << ": " << new_points[i] << " degrees" << endl;
		putText(image, to_string(i + 1), new_points[i], FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 255, 255), 2);
	}
	imshow("test", image);
	return new_points;
}
//5灯排序；并在缺失灯序号处赋值为point（0，0）
vector<Point2f> rank5light_main(Mat input_img, vector<Point2f> input_point)
{
	vector<Point2f> points = input_point;
	// 计算最小包络圆
	Point2f center;
	float radius;
	minEnclosingCircle(points, center, radius);
	// 创建一个绘制图形的图像
	Mat image = input_img.clone();
	// 绘制每个点和最小包络圆
	//for (int i = 0; i < points.size(); i++) 
	//{
	//    circle(image, points[i], 3, Scalar(0, 0, 255), -1);
	//}
	circle(image, center, radius, Scalar(0, 0, 255), 2);

	//vector<float> angles_zero;
	// 输出每个点相对于包络圆圆心的角度信息
	//for (int i = 0; i < points.size(); i++) {
	//    double angle = calculateAngle(center, points[i]);
	//    angles_zero.push_back(angle);
	//    std::cout << "Point " << i + 1 << ": " << angle << " degrees" << std::endl;
	//}

	// 按照角度信息从小到大排序点的向量
	sort(points.begin(), points.end(), [&](const cv::Point& a, const cv::Point& b) { return calculateAngle(center, a) < calculateAngle(center, b); });
	// 输出每个点相对于包络圆圆心的角度信息
	//for (int i = 0; i < points.size(); i++) {
	//    double angle = calculateAngle(center, points[i]);
	//    cout << "Point " << i + 1 << ": " << angle << " degrees" << endl;
	//}

	//绘制多边形
	vector<cv::Point> contour;
	for (int i = 0; i < points.size(); i++) {
		circle(image, points[i], 3, Scalar(0, 0, 255), -1);
		contour.push_back(points[i]);
	}
	cv::polylines(image, contour, true, cv::Scalar(0, 255, 0), 2);
	vector<float> real_angles;
	// 计算每个角并输出
	for (int i = 0; i < contour.size(); i++) {
		int prevIndex = (i == 0) ? contour.size() - 1 : i - 1;
		int nextIndex = (i == contour.size() - 1) ? 0 : i + 1;
		double angle = calculateAngle_2(contour[prevIndex], contour[i], contour[nextIndex]);
		real_angles.push_back(angle);
		//cout<< "Angle " << i + 1 << ": " << angle << " degrees" << endl;
	}
	//判断缺少的灯序号

	vector<vector<float>> model_6light = {
	{ 60,140,70,90,180 },//缺1号
	{ 45,140,70,90,115 },//缺2号
	{ 90,70,140,45,115 },//缺3号
	{ 90,70,140,60,180 },//缺4号
	{ 90,70,140,70,90 },//缺5号
	{ 90,90,90,90,180 },//缺6号
	};
	//vector<float> model1 = { 60,140,70,90,180 };//缺1号
	//vector<float> model2 = { 45,140,70,90,115 };//缺2号
	//vector<float> model3 = { 90,70,140,45,115 };//缺3号
	//vector<float> model4 = { 90,70,140,60,180 };//缺4号
	//vector<float> model5 = { 90,70,140,70,90 };//缺5号
	//vector<float> model6 = { 90,90,90,90,180 };//缺6号

	vector<float> models;//模型匹配临时变量
	vector<vector<float>> deltas(6, std::vector<float>(5, 0));//欧几里得距离：恒定匹配程度，值越小则越接近
	for (int i = 0; i < model_6light.size(); i++)
	{
		vector<float> model_vector = model_6light[i];
		for (int j = 0; j < model_6light[i].size(); j++)
		{
			models = model_vector;
			float delta = calculateEuclideanDistance(models, real_angles);
			deltas[i][j] = delta;
			rotate(model_vector.rbegin(), model_vector.rbegin() + 1, model_vector.rend());
		}
	}
	//输出指标看一下
	//for (int i = 0; i < deltas.size(); i++)
	//{
	//	for (int j = 0; j < deltas[i].size(); j++)
	//	{
	//		cout << "deltasposition: (" << i << ", " << j << ")" <<deltas[i][j]<< std::endl;
	//	}
	//}

	auto min_it = std::min_element(deltas.begin(), deltas.end(),
		[](const std::vector<float>& a, const std::vector<float>& b) {
			return *std::min_element(a.begin(), a.end()) < *std::min_element(b.begin(), b.end());
		});
	// 计算最小值所在的行和列
	int min_row = std::distance(deltas.begin(), min_it);
	int min_col = std::distance((*min_it).begin(), std::min_element((*min_it).begin(), (*min_it).end()));

	std::cout << "Position: (" << min_row << ", " << min_col << ")" << std::endl;
	vector<Point2f> new_points;
	Point2f filler_lack = { 0,0 };
	switch (min_row)
	{
	case 0: //缺1号灯
	{
		switch (min_col)
		{
		case 0: { new_points.push_back(filler_lack); new_points.push_back(points[0]); new_points.push_back(points[3]); new_points.push_back(points[2]); new_points.push_back(points[4]); new_points.push_back(points[1]); }; break;
		case 1: { new_points.push_back(filler_lack); new_points.push_back(points[1]); new_points.push_back(points[4]); new_points.push_back(points[3]); new_points.push_back(points[0]); new_points.push_back(points[2]); }; break;
		case 2: { new_points.push_back(filler_lack); new_points.push_back(points[2]); new_points.push_back(points[0]); new_points.push_back(points[4]); new_points.push_back(points[1]); new_points.push_back(points[3]); }; break;
		case 3: { new_points.push_back(filler_lack); new_points.push_back(points[3]); new_points.push_back(points[1]); new_points.push_back(points[0]); new_points.push_back(points[2]); new_points.push_back(points[4]); }; break;
		case 4: { new_points.push_back(filler_lack); new_points.push_back(points[4]); new_points.push_back(points[2]); new_points.push_back(points[1]); new_points.push_back(points[3]); new_points.push_back(points[0]); }; break;
		}}; break;
	case 1: //缺2号灯
	{
		switch (min_col)
		{
		case 0: { new_points.push_back(points[0]); new_points.push_back(filler_lack); new_points.push_back(points[3]); new_points.push_back(points[2]); new_points.push_back(points[4]); new_points.push_back(points[1]); }; break;
		case 1: { new_points.push_back(points[1]); new_points.push_back(filler_lack); new_points.push_back(points[4]); new_points.push_back(points[3]); new_points.push_back(points[0]); new_points.push_back(points[2]); }; break;
		case 2: { new_points.push_back(points[2]); new_points.push_back(filler_lack); new_points.push_back(points[0]); new_points.push_back(points[4]); new_points.push_back(points[1]); new_points.push_back(points[3]); }; break;
		case 3: { new_points.push_back(points[3]); new_points.push_back(filler_lack); new_points.push_back(points[1]); new_points.push_back(points[0]); new_points.push_back(points[2]); new_points.push_back(points[4]); }; break;
		case 4: { new_points.push_back(points[4]); new_points.push_back(filler_lack); new_points.push_back(points[2]); new_points.push_back(points[1]); new_points.push_back(points[3]); new_points.push_back(points[0]); }; break;
		}}; break;
	case 2: //缺3号灯
	{
		switch (min_col)
		{
		case 0: { new_points.push_back(points[1]); new_points.push_back(points[0]); new_points.push_back(filler_lack); new_points.push_back(points[3]); new_points.push_back(points[4]); new_points.push_back(points[2]); }; break;
		case 1: { new_points.push_back(points[2]); new_points.push_back(points[1]); new_points.push_back(filler_lack); new_points.push_back(points[4]); new_points.push_back(points[0]); new_points.push_back(points[3]); }; break;
		case 2: { new_points.push_back(points[3]); new_points.push_back(points[2]); new_points.push_back(filler_lack); new_points.push_back(points[0]); new_points.push_back(points[1]); new_points.push_back(points[4]); }; break;
		case 3: { new_points.push_back(points[4]); new_points.push_back(points[3]); new_points.push_back(filler_lack); new_points.push_back(points[1]); new_points.push_back(points[2]); new_points.push_back(points[0]); }; break;
		case 4: { new_points.push_back(points[0]); new_points.push_back(points[4]); new_points.push_back(filler_lack); new_points.push_back(points[2]); new_points.push_back(points[3]); new_points.push_back(points[1]); }; break;
		}}; break;
	case 3: //缺4号灯
	{
		switch (min_col)
		{
		case 0: { new_points.push_back(points[1]); new_points.push_back(points[0]); new_points.push_back(points[3]); new_points.push_back(filler_lack); new_points.push_back(points[4]); new_points.push_back(points[2]); }; break;
		case 1: { new_points.push_back(points[2]); new_points.push_back(points[1]); new_points.push_back(points[4]); new_points.push_back(filler_lack); new_points.push_back(points[0]); new_points.push_back(points[3]); }; break;
		case 2: { new_points.push_back(points[3]); new_points.push_back(points[2]); new_points.push_back(points[0]); new_points.push_back(filler_lack); new_points.push_back(points[1]); new_points.push_back(points[4]); }; break;
		case 3: { new_points.push_back(points[4]); new_points.push_back(points[3]); new_points.push_back(points[1]); new_points.push_back(filler_lack); new_points.push_back(points[2]); new_points.push_back(points[0]); }; break;
		case 4: { new_points.push_back(points[0]); new_points.push_back(points[4]); new_points.push_back(points[2]); new_points.push_back(filler_lack); new_points.push_back(points[3]); new_points.push_back(points[1]); }; break;
		}}; break;
	case 4: //缺5号灯
	{
		switch (min_col)
		{
		case 0: { new_points.push_back(points[1]); new_points.push_back(points[0]); new_points.push_back(points[4]); new_points.push_back(points[3]); new_points.push_back(filler_lack); new_points.push_back(points[2]); }; break;
		case 1: { new_points.push_back(points[2]); new_points.push_back(points[1]); new_points.push_back(points[0]); new_points.push_back(points[4]); new_points.push_back(filler_lack); new_points.push_back(points[3]); }; break;
		case 2: { new_points.push_back(points[3]); new_points.push_back(points[2]); new_points.push_back(points[1]); new_points.push_back(points[0]); new_points.push_back(filler_lack); new_points.push_back(points[4]); }; break;
		case 3: { new_points.push_back(points[4]); new_points.push_back(points[3]); new_points.push_back(points[2]); new_points.push_back(points[1]); new_points.push_back(filler_lack); new_points.push_back(points[0]); }; break;
		case 4: { new_points.push_back(points[0]); new_points.push_back(points[4]); new_points.push_back(points[3]); new_points.push_back(points[2]); new_points.push_back(filler_lack); new_points.push_back(points[1]); }; break;
		}}; break;
	case 5: //缺6号灯
	{
		switch (min_col)
		{
		case 0: { new_points.push_back(points[1]); new_points.push_back(points[0]); new_points.push_back(points[3]); new_points.push_back(points[2]); new_points.push_back(points[4]); new_points.push_back(filler_lack); }; break;
		case 1: { new_points.push_back(points[2]); new_points.push_back(points[1]); new_points.push_back(points[4]); new_points.push_back(points[3]); new_points.push_back(points[0]); new_points.push_back(filler_lack); }; break;
		case 2: { new_points.push_back(points[3]); new_points.push_back(points[2]); new_points.push_back(points[0]); new_points.push_back(points[4]); new_points.push_back(points[1]); new_points.push_back(filler_lack); }; break;
		case 3: { new_points.push_back(points[4]); new_points.push_back(points[3]); new_points.push_back(points[1]); new_points.push_back(points[0]); new_points.push_back(points[2]); new_points.push_back(filler_lack); }; break;
		case 4: { new_points.push_back(points[0]); new_points.push_back(points[4]); new_points.push_back(points[2]); new_points.push_back(points[1]); new_points.push_back(points[3]); new_points.push_back(filler_lack); }; break;
		}}; break;
	}

	//输出检测
	for (int i = 0; i < new_points.size(); i++)
	{
		cout << "Angle " << i + 1 << ": " << new_points[i] << " degrees" << endl;
		cv::putText(image, to_string(i + 1), new_points[i], FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 255, 255), 2);
	}

	imshow("test", image);
	
	return new_points;
}
//主灯阵排序：分情况进入不同函数
vector<Point2f> light_rank_main(Mat input_img,vector<Point2f> input_point)
{
	int situation = input_point.size();
	vector<Point2f> return_point;
	Point2f filler = {0,0};
	switch (situation)
	{
	case 6: {return_point = rank6light_main(input_img, input_point); }; break;
	case 5: {return_point = rank5light_main(input_img, input_point); }; break;
	default: {fill_n(return_point.begin(), 6, filler); }; break;
	}
	return return_point;
}


int main() {
	//定义图像分割及圆心定位的返回参数；读入图像
	struct SegmentPoint s1;
    Mat img = imread("D:\\Jupyter\\LOS_rongyu\\fuke\\LOS_rongyu\\1.png", 1);
	
	
	//旋转图像做测试使用
	double angle = 0;
	// 计算旋转中心点
	Point center1(img.cols / 2.0, img.rows / 2.0);
	// 构建旋转矩阵
	Mat rotationMatrix = getRotationMatrix2D(center1, -angle, 1.0);
	//// 应用旋转变换
	Mat rotatedImage;
	warpAffine(img, img, rotationMatrix, img.size());


	//图像分割及圆心定位
	s1 = Segmentation(img, 0.9, 0, 0);
	vector<Point2f> points = s1.imgpoints;
	srand(time(0));

	// 生成 0 到 5 之间的随机整数
	int randomNum = rand() % 6;
	//去除随意一个元素，测试5灯能否正常排序
	points.erase(points.begin() + randomNum);
	/***********************************************************************************/


	vector<Point2f> res;
	//主灯阵排序
	res = light_rank_main(img, points);
	waitKey(0);

    return 0;
}





















//int main()
//{
//	std::vector<cv::Point> points = { cv::Point(100, 100), cv::Point(500, 100), cv::Point(500, 300),
//								 cv::Point(500, 500), cv::Point(100, 500), cv::Point(200, 300) };
//	std::vector<cv::Point> hull;
//	cv::convexHull(points, hull);
//	for (int i = 0; i < hull.size(); i++)
//	{
//		cout << hull[i] << endl;
//	}
//
//	cv::Mat image(600, 600, CV_8UC3, cv::Scalar(255, 255, 255));
//	cv::polylines(image, hull, true, cv::Scalar(0, 0, 255), 2);
//
//	cv::imshow("Convex Hull", image);
//	cv::waitKey(0);
//
//	return 0;
//}