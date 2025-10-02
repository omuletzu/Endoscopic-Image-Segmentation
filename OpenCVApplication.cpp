#include "stdafx.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <fstream>
#include <iostream>
#include <random>
#include <float.h>
#include <vector>
#include <time.h>
#include <map>

using namespace std;
using namespace cv;

const int IMAGE_DIMENSION = 256;

struct ImagePair {
	string original_img_path;
	string mask_img_path;
};

bool isinside(const Mat& img, double i, double j) {
	return i >= 0 && j >= 0 && i < img.rows && j < img.cols;
}

vector<ImagePair> get_all_image_pairs(string folder_path, bool mask_exist) {
	vector<ImagePair> image_pair_vec;

	string search_path = folder_path + "*.*";

	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);

	if (hFind == INVALID_HANDLE_VALUE) {
		return image_pair_vec;
	}

	do {
		if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
			string filename = fd.cFileName;

			if (filename.find("watershed_mask") != string::npos) {
				continue;
			}

			string ext = filename.substr(filename.find_last_of(".") + 1);

			if (ext == "bmp") {
				string full_path = folder_path + filename;

				if (mask_exist) {
					string mask_full_path = folder_path + filename.substr(0, filename.length() - 4) + "_watershed_mask.bmp";
					image_pair_vec.push_back({ full_path, mask_full_path });
				}
				else {
					image_pair_vec.push_back({ full_path, ""});
				}
			}
		}
	} while (::FindNextFile(hFind, &fd));

	::FindClose(hFind);

	return image_pair_vec;
}

Mat_<uchar> convert_vec3bmask_to_uchar(Mat_<Vec3b> img) {	// R = G = B for mask
	Mat_<uchar> new_img(img.size());
	new_img.setTo(0);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			new_img(i, j) = img(i, j)[0];
		}
	}

	return new_img;
}

Mat_<Vec3b> convert_hsv(Mat_<Vec3b> img) {
	Mat_<Vec3b> result(img.rows, img.cols);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			float r = img(i, j)[2] / 255.0;
			float g = img(i, j)[1] / 255.0;
			float b = img(i, j)[0] / 255.0;

			float maxx = max(r, max(g, b));
			float minn = min(r, min(g, b));

			float c = maxx - minn;

			float v = maxx;

			float s = 0;

			if (v != 0) {
				s = c / v;
			}

			float h = 0;

			if (c != 0) {
				if (maxx == r) {
					h = 60 * ((g - b) / c);
				}

				if (maxx == g) {
					h = 120 + 60 * ((b - r) / c);
				}

				if (maxx == b) {
					h = 240 + 60 * ((r - g) / c);
				}
			}

			if (h < 0) {
				h += 360;
			}

			result(i, j)[0] = h * 255 / 360;
			result(i, j)[1] = s * 255;
			result(i, j)[2] = v * 255;
		}
	}

	return result;
}

Mat_<Vec3b> reverse_hvs(Mat_<Vec3b> img) {
	Mat_<Vec3b> result(img.rows, img.cols);

	cvtColor(img, result, COLOR_HSV2BGR_FULL);

	return result;
}

Vec3b bilinear_interpolate(Mat_<Vec3b> img, float src_x, float src_y) {
	int x0 = static_cast<int>(floor(src_x));
	int y0 = static_cast<int>(floor(src_y));

	int x1 = min(x0 + 1, img.cols - 1);
	int y1 = min(y0 + 1, img.rows - 1);

	float dx = src_x - x0;
	float dy = src_y - y0;

	Vec3b result;

	for (int k = 0; k < 3; k++) {
		float a = img(y0, x0)[k];
		float b = img(y0, x1)[k];
		float c = img(y1, x0)[k];
		float d = img(y1, x1)[k];

		float interpol =
			a * (1 - dx) * (1 - dy) +
			b * dx * (1 - dy) +
			c * (1 - dx) * dy +
			d * dx * dy;
		
		interpol = min(interpol, 255.0f);
		interpol = max(interpol, 0.0f);

		result[k] = static_cast<uchar>(interpol);
	}

	return result;
}

//Mat_<Vec3b> resize_image_original(Mat_<Vec3b> img, int new_size) {
//	Mat resized_img;
//	resize(img, resized_img, Size(new_size, new_size), INTER_LINEAR);
//	return resized_img;
//}

Mat_<Vec3b> resize_image_original(Mat_<Vec3b> img, int new_size) {
	Mat_<Vec3b> new_image(new_size, new_size);

	double scale_y = static_cast<double>(img.rows) / new_size;
	double scale_x = static_cast<double>(img.cols) / new_size;

	for (int i = 0; i < new_image.rows; i++) {
		for (int j = 0; j < new_image.cols; j++) {
			double src_x = j * scale_x;
			double src_y = i * scale_y;
			new_image(i, j) = bilinear_interpolate(img, src_x, src_y);
		}
	}

	return new_image;
}

Mat_<Vec3b> resize_image_mask(Mat_<Vec3b> img, int new_size) {
	Mat resized_img;
	resize(img, resized_img, Size(new_size, new_size), INTER_NEAREST);
	return resized_img;
}

//Mat_<Vec3b> resize_image_mask(Mat_<Vec3b> img, int new_size) {
//	Mat_<Vec3b> new_image(new_size, new_size);
//
//	double scale_y = static_cast<double>(img.rows) / new_size;
//	double scale_x = static_cast<double>(img.cols) / new_size;
//
//	for (int i = 0; i < new_image.rows; i++) {
//		for (int j = 0; j < new_image.cols; j++) {
//			int src_x = static_cast<int>(round(j * scale_x));
//			int src_y = static_cast<int>(round(i * scale_y));
//
//			src_x = min(src_x, img.cols - 1);
//			src_y = min(src_y, img.rows - 1);
//
//			new_image(i, j) = img(src_y, src_x);
//		}
//	}
//
//	return new_image;
//}

void resize_all_images(vector<ImagePair> imagePair) {
	for (auto img : imagePair) {
		String file_original = img.original_img_path;
		String file_mask = img.mask_img_path;

		try {
			Mat_<Vec3b> img_original = imread(file_original);
			Mat_<Vec3b> img_mask = imread(file_mask);

			imwrite(file_original, resize_image_original(img_original, IMAGE_DIMENSION));

			Mat_<Vec3b> img_mask_resized = resize_image_mask(img_mask, IMAGE_DIMENSION);
			Mat_<uchar> single_channel_mask = convert_vec3bmask_to_uchar(img_mask_resized);

			/*Mat single_channel_mask;
			cvtColor(img_mask_resized, single_channel_mask, COLOR_BGR2GRAY);*/

			imwrite(file_mask, single_channel_mask);
		}
		catch (const exception& e) {}
	}
}

//Mat_<Vec3b> rotate_image(Mat_<Vec3b> img, int angle, bool is_mask) {
//	Point2f center(img.cols / 2.0f, img.rows / 2.0f);
//	Mat rot_mat = getRotationMatrix2D(center, angle, 1.0);
//	Mat rot_img;
//
//	int interpolation = is_mask ? INTER_NEAREST : INTER_LINEAR;
//
//	warpAffine(img, rot_img, rot_mat, img.size(), interpolation, BORDER_CONSTANT, Scalar(0, 0, 0));
//
//	return rot_img;
//}

Mat_<Vec3b> rotate_image(Mat_<Vec3b> img, int angle, bool is_mask) {
	Mat_<Vec3b> rot_img(img.rows, img.cols, Vec3b(0, 0, 0));
	
	double angle_rad = angle * CV_PI / 180.0f;
	double cos_a = cos(angle_rad);
	double sin_a = sin(angle_rad);

	double cx = img.cols / 2.0f;
	double cy = img.rows / 2.0f;

	for (int i = 0; i < rot_img.rows; i++) {
		for (int j = 0; j < rot_img.cols; j++) {
			double x = j - cx;
			double y = i - cy;

			double src_x = cos_a * x - sin_a * y + cx;
			double src_y = sin_a * x + cos_a * y + cy;

			if (isinside(img, src_y, src_x)) {

				if (is_mask) {
					int x_n = static_cast<int>(round(src_x));
					int y_n = static_cast<int>(round(src_y));

					x_n = min(max(0, x_n), img.cols - 1);
					y_n = min(max(0, y_n), img.rows - 1);

					rot_img(i, j) = img(y_n, x_n);
				}
				else {
					rot_img(i, j) = bilinear_interpolate(img, src_x, src_y);
				}
			}
		}
	}

	return rot_img;
}

//Mat_<Vec3b> flip_image(Mat_<Vec3b> img, int vert_horiz) {
//	Mat_<Vec3b> flip_img;
//	flip(img, flip_img, vert_horiz);
//	return flip_img;
//}

Mat_<Vec3b> flip_image(Mat_<Vec3b> img, int vert_horiz) {
	Mat_<Vec3b> flip_img(img.rows, img.cols, Vec3b(0, 0, 0));
	
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int src_i = i;
			int src_j = j;

			if (vert_horiz) {
				src_j = img.cols - 1 - j;
			}
			else {
				src_i = img.rows - 1 - i;
			}

			flip_img(i, j) = img(src_i, src_j);
		}
	}

	return flip_img;
}

//Mat_<Vec3b> zoom_image(Mat_<Vec3b> img, double zoom_factor, bool is_mask) {
//	int height = static_cast<int>(img.rows * zoom_factor);
//	int width = static_cast<int>(img.cols * zoom_factor);
//
//	int interpolation = is_mask ? INTER_NEAREST : INTER_LINEAR;
//
//	Mat zoomed_img;
//	resize(img, zoomed_img, Size(width, height), 0, 0, interpolation);
//
//	Mat resized_result;
//
//	if (zoom_factor > 1) {
//		int x_cord = (zoomed_img.cols - IMAGE_DIMENSION) / 2;
//		int y_cord = (zoomed_img.rows - IMAGE_DIMENSION) / 2;
//
//		Rect crop(x_cord, y_cord, IMAGE_DIMENSION, IMAGE_DIMENSION);
//
//		resized_result = zoomed_img(crop);
//	}
//	else {
//		resized_result = zoomed_img;
//		copyMakeBorder(
//			resized_result, 
//			resized_result, 
//			0, 
//			IMAGE_DIMENSION - resized_result.rows, 
//			0, 
//			IMAGE_DIMENSION - resized_result.cols, 
//			BORDER_CONSTANT, 
//			Scalar(0, 0, 0));
//	}
//
//	return resized_result;
//}

Mat_<Vec3b> zoom_image(Mat_<Vec3b> img, double zoom_factor, bool is_mask) {
	int height = static_cast<int>(img.rows * zoom_factor);
	int width = static_cast<int>(img.cols * zoom_factor);

	Mat_<Vec3b> zoomed_img(height, width, Vec3b(0, 0, 0));

	double scale_y = static_cast<double>(img.rows) / height;
	double scale_x = static_cast<double>(img.cols) / width;

	for (int i = 0; i < zoomed_img.rows; i++) {
		for (int j = 0; j < zoomed_img.cols; j++) {
			double src_x = j * scale_x;
			double src_y = i * scale_y;

			if (is_mask) {
				int x_n = static_cast<int>(round(src_x));
				int y_n = static_cast<int>(round(src_y));

				x_n = min(max(0, x_n), img.cols - 1);
				y_n = min(max(0, y_n), img.rows - 1);

				zoomed_img(i, j) = img(y_n, x_n);
			}
			else {
				zoomed_img(i, j) = bilinear_interpolate(img, src_x, src_y);
			}
		}
	}

	Mat_<Vec3b> zoomed_img_final;

	if (zoom_factor > 1.0) {
		int x_crop = (width - IMAGE_DIMENSION) / 2;
		int y_crop = (height - IMAGE_DIMENSION) / 2;

		Rect roi(x_crop, y_crop, IMAGE_DIMENSION, IMAGE_DIMENSION);
		zoomed_img_final = zoomed_img(roi).clone();
	}
	else {
		zoomed_img_final = Mat_<Vec3b>(IMAGE_DIMENSION, IMAGE_DIMENSION, Vec3b(0, 0, 0));

		int x_offset = (IMAGE_DIMENSION - zoomed_img.cols) / 2;
		int y_offset = (IMAGE_DIMENSION - zoomed_img.rows) / 2;

		for (int i = 0; i < zoomed_img.rows; i++) {
			for (int j = 0; j < zoomed_img.cols; j++) {
				zoomed_img_final(i + y_offset, j + x_offset) = zoomed_img(i, j);
			}
		}
	}

	return zoomed_img_final;
}

//Mat_<Vec3b> bright_image(Mat_<Vec3b> img, double bright_factor) {
//	Mat_<Vec3b> bright_img;
//	img.convertTo(bright_img, -1, 1, bright_factor);
//	return bright_img;
//}

Mat_<Vec3b> bright_image(Mat_<Vec3b> img, double bright_factor) {
	Mat_<Vec3b> bright_img(img.rows, img.cols);

	for (int i = 0; i < bright_img.rows; i++) {
		for (int j = 0; j < bright_img.cols; j++) {

			for (int k = 0; k < 3; k++) {
				int value = static_cast<int>(img(i, j)[k] * bright_factor);
				value = max(0, min(value, 255));

				bright_img(i, j)[k] = static_cast<uchar>(value);
			}
		}
	}

	return bright_img;
}

//Mat_<Vec3b> contrast_image(Mat_<Vec3b> img, double contrast_factor) {
//	Mat_<Vec3b> contrast_img;
//	img.convertTo(contrast_img, -1, contrast_factor, 0);
//	return contrast_img;
//}

Mat_<Vec3b> contrast_image(Mat_<Vec3b> img, double contrast_factor) {
	Mat_<Vec3b> contrast_img(img.rows, img.cols);

	for (int i = 0; i < contrast_img.rows; i++) {
		for (int j = 0; j < contrast_img.cols; j++) {

			for (int k = 0; k < 3; k++) {
				int value = static_cast<int>((img(i, j)[k] - 128) * contrast_factor + 128);
				value = max(0, min(value, 255));

				contrast_img(i, j)[k] = static_cast<uchar>(value);
			}
		}
	}

	return contrast_img;
}

//Mat_<Vec3b> saturate_image(Mat_<Vec3b> img, double saturate_factor) {
//	Mat hsv_img;
//
//	cvtColor(img, hsv_img, COLOR_BGR2HSV);
//
//	vector<Mat> hsv_channel(3);
//	split(hsv_img, hsv_channel);
//
//	hsv_channel[1] *= saturate_factor;
//
//	threshold(hsv_channel[1], hsv_channel[1], 255, 255, THRESH_TRUNC);
//
//	merge(hsv_channel, hsv_img);
//
//	Mat_<Vec3b> final_img;
//
//	cvtColor(hsv_img, final_img, COLOR_HSV2BGR);
//
//	return final_img;
//}

Mat_<Vec3b> saturate_image(Mat_<Vec3b> img, double saturate_factor) {
	Mat_<Vec3b> hsv_img = convert_hsv(img);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int s = static_cast<int>(hsv_img(i, j)[1] * saturate_factor);
			s = min(s, 255);
			s = max(s, 0);

			hsv_img(i, j)[1] = static_cast<uchar>(s);
		}
	}

	return reverse_hvs(hsv_img);
}

void random_augment(vector<ImagePair> imagePair) {
	for (auto img : imagePair) {
		String file_original = img.original_img_path;
		String file_mask = img.mask_img_path;
 
		try {
			Mat_<Vec3b> img_original = imread(file_original);
			Mat_<Vec3b> img_mask = imread(file_mask);

			int option = rand() % 3;

			if (option == 0) {	// ROTATE
				int angles[] = { 30, 60, 90, 180 };
				int rnd_index = rand() % (sizeof(angles) / sizeof(int));

				cout << "bingo " << angles[rnd_index] << " " << file_original << "\n";

				img_original = rotate_image(img_original, angles[rnd_index], false);	// false - original
				img_mask = rotate_image(img_mask, angles[rnd_index], true);	// true - mask
			}
			else if (option == 1) {		// FLIP
				int vert_horiz = rand() % 2;	// 0 - orizontal, 1 - vertical
				
				img_original = flip_image(img_original, vert_horiz);
				img_mask = flip_image(img_mask, vert_horiz);
			}
			else if (option == 2) {		// ZOOM
				double zoom_factor = 0.8 + 0.4 * (rand() / static_cast<double>(RAND_MAX));	// 0.8 - 1.2

				img_original = zoom_image(img_original, zoom_factor, false);	// false - original
				img_mask = zoom_image(img_mask, zoom_factor, true);	// true - mask
			}

			option = 3 + rand() % 3;

			if (option == 3) {	// BRIGHTEN
				double rnd_bright_factor = -30 + (rand() / (double)RAND_MAX) * 60;	// -30, 30
				img_original = bright_image(img_original, rnd_bright_factor);
			}
			else if (option == 4) {		// CONTRAST
				double rnd_contrast_factor = 0.8 + 0.4 * (rand() / static_cast<double>(RAND_MAX));	// 0.8 - 1.2
				img_original = contrast_image(img_original, rnd_contrast_factor);
			}
			else if (option == 5) {		// SATURATION
				double rnd_saturate_factor = 0.8 + 0.4 * (rand() / static_cast<double>(RAND_MAX));	// 0.8 - 1.2
				img_original = saturate_image(img_original, rnd_saturate_factor);
			}

			string aug_file_original = "IMAGES/dataset_aug/" + file_original.substr(file_original.find_last_of("/") + 1);
			string aug_file_mask = "IMAGES/dataset_aug/" + file_mask.substr(file_mask.find_last_of("/") + 1);

			imwrite(aug_file_original, resize_image_original(img_original, IMAGE_DIMENSION));
			imwrite(aug_file_mask, resize_image_mask(img_mask, IMAGE_DIMENSION));
		}
		catch (const exception& e) {}
	}
}

void color_image_class_id(vector<ImagePair> imagePair) {
	map<int, Vec3b> color_map;
	
	color_map[0] = Vec3b(127, 127, 127);
	color_map[1] = Vec3b(140, 140, 210);
	color_map[2] = Vec3b(114, 114, 255);
	color_map[3] = Vec3b(156, 70, 231);
	color_map[4] = Vec3b(75, 183, 186);
	color_map[5] = Vec3b(0, 255, 170);
	color_map[6] = Vec3b(0, 85, 255);
	color_map[7] = Vec3b(0, 0, 255);
	color_map[8] = Vec3b(0, 255, 255);
	color_map[9] = Vec3b(184, 255, 169);
	color_map[10] = Vec3b(165, 160, 255);
	color_map[11] = Vec3b(128, 50, 0);
	color_map[12] = Vec3b(0, 74, 111);

	for (auto img : imagePair) {
		Mat_<uchar> segmented_image_class_id = imread(img.original_img_path, IMREAD_GRAYSCALE);

		Mat_<Vec3b> segmented_image(IMAGE_DIMENSION, IMAGE_DIMENSION);

		for (int i = 0; i < segmented_image_class_id.rows; i++) {
			for (int j = 0; j < segmented_image_class_id.cols; j++) {
				segmented_image(i, j) = color_map[segmented_image_class_id(i, j)];
			}
		}

		string final_filename = "IMAGES/final_segmented_images/" + img.original_img_path.substr(img.original_img_path.find_last_of("/") + 1);

		resize(segmented_image, segmented_image, Size(854, 480), 0, 0, INTER_NEAREST);

		imwrite(final_filename, segmented_image);
	}
}

int main() {
	srand(time(NULL));

	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_FATAL);

	vector<ImagePair> img_vec = get_all_image_pairs("IMAGES_2/", true);

	//resize_all_images(img_vec);
	//random_augment(img_vec);
	//color_image_class_id(img_vec);

	return 0;
}