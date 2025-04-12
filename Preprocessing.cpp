#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <filesystem>
#include<fstream>
using namespace cv;
using namespace std;
namespace fs = filesystem;

void SaveAsBinary(const string& filename, const Mat& image)
{
	ofstream Writer(filename, ios::binary);
	if (!Writer)
	{
		cout << "Could Not Open File" << endl;
		return;
	}
	Writer.write(reinterpret_cast<const char*>(image.data), image.total() * image.elemSize());//Write image data to binary file
	Writer.close();
	cout << "File Created Successfully" << endl;
}

int main()
{
	fs::directory_iterator Start{ "train" };//Take Images from folder "/train"
	fs::directory_iterator End{};
	auto Iter{ Start };
	string imagepath;
	Mat img;
	for (Iter; Iter != End; Iter++)//Loop for saving training data to binary
	{
		imagepath = Iter->path().string();
		img = imread(imagepath, IMREAD_GRAYSCALE);//Open Image as Grayscale
		imagepath = imagepath.substr(5, imagepath.size() - 9);
		imagepath += ".bin";//Change Image type to bin
		resize(img, img, Size(224, 224), 0, 0, INTER_AREA);//Resize Image to 224x224
		img.convertTo(img, CV_32F, 1.0 / 255);//Normalise pixel values
		SaveAsBinary("normalisedtrain/"+imagepath, img);//Save Image as binary in folder "/normalisedtrain"
	}
	Start = fs::directory_iterator("val");//Take Images from folder "/val"
	Iter = Start;
	for (Iter; Iter != End; Iter++)//Loop for saving validation data to binary
	{
		imagepath = Iter->path().string();
		img = imread(imagepath, IMREAD_GRAYSCALE);//Open Image as Grayscale
		imagepath = imagepath.substr(3, imagepath.size() - 7);
		imagepath += ".bin";//Change Image type to bin
		resize(img, img, Size(224, 224), 0, 0, INTER_AREA);//Resize Image to 224x224
		img.convertTo(img, CV_32F, 1.0 / 255);//Normalise pixel values
		SaveAsBinary("normalisedval/" + imagepath, img);//Save Image as binary in folder "/normalisedval"
	}
	return 0;
}

