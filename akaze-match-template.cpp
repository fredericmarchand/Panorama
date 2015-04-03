#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

#define DEBUG 0

using namespace std;
using namespace cv;

const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

int main(void)
{
	puts("opening");
    Mat img1 = imread("keble_a_half.bmp", IMREAD_GRAYSCALE);
    Mat img2 = imread("keble_b_long.bmp", IMREAD_GRAYSCALE);
	Mat img3 = Mat(img2.rows, img2.cols, CV_8UC1);
	string final_merged_output_name = "merged.jpg";
	string final_warped_output_name = "warped.jpg";
	//img2.copyTo(img3);

    Mat homography;

    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
	puts("Have opened");

    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
    akaze->detectAndCompute(img2, noArray(), kpts2, desc2);

	puts("have commputed akaze");

    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);
	puts("Have done match");

    vector<Point2f> matched1, matched2;
	vector<Point2f> inliers1, inliers2;
 
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;

        if(dist1 < nn_match_ratio * dist2) {
            matched1.push_back(kpts1[first.queryIdx].pt);
			matched2.push_back(kpts2[first.trainIdx].pt);
        }
    }
	printf("Matches %d %d\n", matched1.size(), matched2.size());

	homography = findHomography(matched1, matched2, RANSAC);

#if DEBUG == 1
	cout << homography << endl;
#endif

	warpPerspective(img1, img3, homography, img3.size());

#if DEBUG == 1
	Mat img_matches;
	drawMatches( img1, kpts1, img2, kpts2, nn_matches, img_matches );
	imshow("Input4",img_matches);
#endif

	for (int i = 0; i < img3.size().height; ++i) 
	{
		for (int j = 0; j < img3.size().width; ++j)
		{
			img2.at<uchar>(i, j) = (img2.at<uchar>(i, j) | img3.at<uchar>(i, j));
		}
	}

#if DEBUG == 1
	medianBlur(img2, img3, 3);
#endif

    //Display input and output
    //imshow("Input1",img1);
    imshow("Merged", img2);
	imshow("Warped", img3);
	imwrite(final_merged_output_name, img2);
	imwrite(final_warped_output_name, img3);
	waitKey(0);

    return 0;
}