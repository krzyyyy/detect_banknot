//============================================================================
// Name        : OpenCV_cube.cpp
// Author      : Krzysztof Majda
//============================================================================

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;
//funkcje kolejno rotacji punktów 3d, obliczania środkow każdej ze scian i obslugi przyciskow
void mul(const Mat& src, vector<Vec3f> pts, vector<Vec3f>& out );
void mean(const vector<Vec3f>& points, vector<vector<size_t>>& id_img, vector<Vec3f>& out );
void key_handling(int sign, Vec3f& rot_mat, Vec3f& trans);

vector <Mat> images(0);// obrazy na ścianach kostki
vector <int> band({-100,100}); // skladowe wspolzendnych punktow 3d

vector <Vec3f> points_3d, points_mean_3d;//kolejno wierzcholki kostki i punkty przeciecia przekatnych dla kazdej ze scian
vector <Vec2f> points_2d, points_mean_2d;// jw. w 2d
vector <vector<size_t>> id_img_points(0), id_img_points_good(0);// wektor indeksow punktow dla kazdej ze scian
//parametry dla funkcji projectPoints
double focal_length = 150;// ogniskowa
Point2d center(0, 0);// miejsce poczatku ukladu wspulrzednych obrazu
Mat camera_matrix = (Mat_<double>(3,3) << focal_length, 0, center.x,
										0 , focal_length, center.y, 0, 0, 1);
Mat dist_coeffs = Mat::zeros(4,1,cv::DataType<double>::type);// info o znieksztalceniu obrazu przez soczewke itd. - brak
Vec3f trans(0, 0, -600);// odleglosc od obiektu
int sign =0; // nr. znaku
Mat rot_mat = Mat::eye(3,  3, CV_32FC1); // macierz rotacji

int main() {
	for(auto& x : band)//generowaie wierzcholkow kostki w 3d
		for(auto y:band)
			for(auto z:band)
				points_3d.push_back(Vec3f(x,y,z));
	for(int id =0;id<3;id++)// generowanie wektorow ideksow wierzcholkow dla kazdej ze scian
		for(auto b : band){
			vector <size_t> temp;
			for(size_t i =0;i<points_3d.size();i++){
				if(points_3d[i][id]==b)
					temp.push_back(i);
			}
			id_img_points.push_back(temp);
		}
	for(int i=1;i<7;i++){ // wczytywanie obrazow
		stringstream name;
		name<<i;
		images.push_back(imread(name.str()+".png"));
	}
	vector <Vec2f> corner_points({Vec2f(0,0),Vec2f(images[0].cols,0),// wierzcholki obrazow
								  Vec2f(0,images[0].rows),Vec2f(images[0].cols,images[0].rows)} );

	while(sign!=27){
		Vec3f rot(0, 0, 0);
		Mat img(400, 400, CV_8UC3, Scalar(0, 0, 0));
		putText(img, "X: w, a  Y: e, s  Z: r, d Zoom: p, l" , Point(10,20),
				FONT_HERSHEY_SIMPLEX , 0.5, Scalar(20,220,40));
		key_handling(sign, rot, trans);// obsluga klawiatury
		Rodrigues(rot, rot_mat);// tworzenie macerzy rotacji

		mul(rot_mat, points_3d, points_3d);// rotacja punktuw 3d
		mean(points_3d, id_img_points, points_mean_3d);//obliczanie srodkow dla kazdej sciany

		//kolejno projekcja wierzcholkow szecianu i srodkow scian na obraz 2d
		projectPoints(points_3d, Vec3f(0, 0, 0), trans, camera_matrix, dist_coeffs, points_2d);
		projectPoints(points_mean_3d, Vec3f(0, 0, 0), trans, camera_matrix, dist_coeffs, points_mean_2d);


//		for(auto p:points_2d)//zaznaczanie wierzcholkow na obrazie
//			drawMarker(img, Point(p+Vec2f(200,200)), Scalar(200,0,120));
//		for(auto p:points_mean_2d)//zaznaczanie srodkow scian na obrazie
//			drawMarker(img, Point(p+Vec2f(200,200)), Scalar(50,0,220));
		vector <size_t> indexes_good(0);// indeksy scian ktore będą wyświetlane
		vector <size_t> indexes_of_id({0, 1, 2, 3, 4, 5});// indeksy scian
		sort(indexes_of_id.begin(), indexes_of_id.end(),//sortowanie indeksow scian na podstawie skladowej Z srodkow tych scian
				[points_mean_3d](size_t a, size_t b){return points_mean_3d[a][2]>points_mean_3d[b][2];});
		id_img_points_good.clear();
		//Wybieranie scian do wyswietlenia tak ze srodki tych scian na obrazie 2d nie zawieraja sie
		//wewnatrz blizszej sciany( tj. takiej dla ktorej skladowa Z srodka ma wieksza wartosc)
		for(auto id:indexes_of_id){
			bool if_internal = false;
			for(auto pts:id_img_points_good){
				vector <Vec2f> area(4),area2(0);
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

//		for(auto p:id_img_points_good)//wyswietlanie widocznych wierzchlkow
//			for(auto pt:p)
//				drawMarker(img, Point(points_2d[pt]+Vec2f(200,200)), Scalar(100, 200,10));

		for(auto id:indexes_good){// wyswietlanie widocznych obrazow
			vector <Vec2f> wall_points(4);
			generate(wall_points.begin(), wall_points.end(),//generowanie punktow 2d
						[id, id_img_points, points_2d, n=0]()mutable{return points_2d[id_img_points[id][n++]]+Vec2f(200,200);});
			Mat mat_transf = getPerspectiveTransform(corner_points, wall_points);//obliczanie macierzy transformacji
			Mat temp;
			warpPerspective(images[id], temp, mat_transf, Size(img.rows, img.cols));//transformacja obrazu
			add(temp, img, img);

		}
		imshow("okno", img);
		sign = waitKey(0);
	}
	return 0;
}
void mul(const Mat& src, vector<Vec3f> pts, vector<Vec3f>& out ){
	out.clear();
	for(auto pt:pts)
		out.push_back(Vec3f(((float*)Mat(src*Mat(pt)).data)));
}
void mean(const vector<Vec3f>& points, vector<vector <size_t>>& id_img, vector<Vec3f>& out ){
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
void key_handling(int sign, Vec3f& rot_mat, Vec3f& trans){
	if(sign == 'p'&&trans[2]<-400)
		trans[2]+=2;
	else if(sign == 'l'&&trans[2]>-800)
		trans[2]-=2;
	else if(sign == 'w')
		rot_mat[0]=0.01;
	else if(sign == 'e')
		rot_mat[1]=0.01;
	else if(sign == 'r')
		rot_mat[2]=0.01;
	else if(sign == 'a')
		rot_mat[0]=-0.01;
	else if(sign == 's')
		rot_mat[1]=-0.01;
	else if(sign == 'd')
		rot_mat[2]=-0.01;
}
