//============================================================================
// Name        : OpenCV_cube.cpp
// Author      : Krzysztof Majda
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

//void multiply

int banding[] = {-100,100};
vector <int> band(banding, banding+2);

vector <Vec3f> points_3d, points_mean_3d;
vector <Vec2f> points_2d, points_mean_2d;
vector <vector<int>> id_img_points(0), id_img_points_good(0);
double focal_length = 100;
Point2d center(0, 0);
Mat camera_matrix = (Mat_<double>(3,3) << focal_length, 0, center.x,
										0 , focal_length, center.y, 0, 0, 1);
Mat dist_coeffs = Mat::zeros(4,1,cv::DataType<double>::type);
Vec3f trans(0, 0, -800);
int sign =0;
Mat rot_mat = Mat::eye(3,  3, CV_32FC1);

void mul(const Mat& src, vector<Vec3f> pts, vector<Vec3f>& out );
void mean(const vector<Vec3f>& points, vector<vector<int>>& id_img, vector<Vec3f>& out );
//void find_visiable()
int main() {
	Mat img1 = imread("marcin.png");
	vector <Vec2f> corner_points({Vec2f(0,0),Vec2f(img1.cols,0), Vec2f(0,img1.rows),Vec2f(img1.cols,img1.rows)} );
	for(auto& x : band)
		for(auto y:band)
			for(auto z:band)
				points_3d.push_back(Vec3f(x,y,z));
	for(int id =0;id<3;id++)
		for(auto b : band){
			vector <int> temp;
			for(int i =0;i<points_3d.size();i++){
				if(points_3d[i][id]==b)
					temp.push_back(i);
			}
			id_img_points.push_back(temp);
		}

	while(sign!=27){
		Vec3f rot(0, 0, 0);
		Mat img(400, 400, CV_8UC3, Scalar(0, 0, 0));
		if(sign == 'p')
			trans[2]+=1;
		else if(sign == 'w')
			rot[0]=0.01;
		else if(sign == 'e')
			rot[1]=0.01;
		else if(sign == 'r')
			rot[2]=0.01;
		Rodrigues(rot, rot_mat);

		mul(rot_mat, points_3d, points_3d);
		mean(points_3d, id_img_points, points_mean_3d);


		projectPoints(points_3d, Vec3f(0, 0, 0), trans, camera_matrix, dist_coeffs, points_2d);
		projectPoints(points_mean_3d, Vec3f(0, 0, 0), trans, camera_matrix, dist_coeffs, points_mean_2d);


		for(auto p:points_2d)
			drawMarker(img, Point(p+Vec2f(200,200)), Scalar(200,0,120));
		for(auto p:points_mean_2d)
			drawMarker(img, Point(p+Vec2f(200,200)), Scalar(50,0,220));
		vector <size_t> indexes_good(0);
		vector <size_t> indexes_of_id({0, 1, 2, 3, 4, 5});//iota nie dziala na tym kompilatorze:)
		sort(indexes_of_id.begin(), indexes_of_id.end(),
				[&points_mean_3d](size_t a, size_t b){return points_mean_3d[a][2]>points_mean_3d[b][2];});
		id_img_points_good.clear();
		for(auto id:indexes_of_id){
			bool if_internal = false;
			for(auto pts:id_img_points_good){
				vector <Vec2f> area(4);
				vector <Vec2f> area2(0);
				generate(area.begin(), area.end(),[points_2d, pts, n=0]()mutable{return points_2d[pts[n++]];});
				convexHull(area, area2);
				if(pointPolygonTest(area2, points_mean_2d[id], false)>=0)
					if_internal=true;
			}
			if(if_internal==false){
				id_img_points_good.push_back(id_img_points[id]);
				indexes_good.push_back(id);
			}
		}

		for(auto p:id_img_points_good)
			for(auto pt:p)
				drawMarker(img, Point(points_2d[pt]+Vec2f(200,200)), Scalar(100, 200,10));

		for(auto id:indexes_good){
			vector <Vec2f> wall_points(4);
			generate(wall_points.begin(), wall_points.end(),
						[id, id_img_points, points_2d, n=0]()mutable{return points_2d[id_img_points[id][n++]]+Vec2f(200,200);});
			Mat mat_transf = getPerspectiveTransform(corner_points, wall_points);
			warpPerspective(img1, img, mat_transf, Size(img1.rows, img1.cols));
		}

		imshow("okno", img);
		sign = waitKey(0);

	}

	return 0;
}
void mul(const Mat& src, vector<Vec3f> pts, vector<Vec3f>& out ){
	out.clear();
	for(auto pt:pts){
		out.push_back(Vec3f(((float*)Mat(src*Mat(pt)).data)));
	}
}
void mean(const vector<Vec3f>& points, vector<vector <int>>& id_img, vector<Vec3f>& out ){
	out.clear();
	for(auto x:id_img){
		Vec3f sum(0, 0, 0);
		for(auto id:x){
			sum+=points[id];
		}
		sum/=4;
		out.push_back(sum);
	}
}
