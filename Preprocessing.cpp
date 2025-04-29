#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <filesystem>
#include<fstream>
using namespace cv;
using namespace std;
namespace fs = filesystem;

#define RESIZEDIM 224
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
}

struct bboxcoordinate
{
	float label;
	float center_x;
	float center_y;
	float width;
	float height;
};

void UpdateBBoxCoordinates(string filename,int new_width,int new_height,int paddingx,int paddingy)
{
	ifstream reader(filename);
	if (!reader)
	{
		cout << "Could Not Open File" << endl;
		return;
	}
	vector<bboxcoordinate> coords;
	float temp;
	while (reader >> temp)//Read old coordinates
	{
		coords.push_back(bboxcoordinate());
		coords.at(coords.size() - 1).label = temp;
		reader >> temp;
		coords.at(coords.size() - 1).center_x = temp;
		reader >> temp;
		coords.at(coords.size() - 1).center_y = temp;
		reader >> temp;
		coords.at(coords.size() - 1).width = temp;
		reader >> temp;
		coords.at(coords.size() - 1).height = temp;
	}
	reader.close();

	string newfilename;
	if (filename.substr(0, 5) == "train")
		newfilename = "normalisedtrainlabels/" + filename.substr(12, filename.size());
	else
		newfilename = "normalisedvallabels/" + filename.substr(10, filename.size());
	ofstream writer(newfilename);
	for (int i = 0; i < coords.size(); i++)//write updated coordinates to a new file in new folder
	{
		writer << coords.at(i).label << ' ';
		writer << (coords.at(i).center_x * new_width + paddingx) / RESIZEDIM << ' ';
		writer <<  (coords.at(i).center_y * new_height + paddingy) / RESIZEDIM << ' ';
		writer << (coords.at(i).width * new_width) / RESIZEDIM << ' ';
		writer << (coords.at(i).height * new_height) / RESIZEDIM << '\n';
	}
	writer.close();
}

int countBBox(string filename)
{
	ifstream reader(filename);
	string temp;
	int count = 0;
	while (getline(reader, temp))
		count++;
	reader.close();
	return count;
}

Mat resizeImageAndUpdateBBox(Mat& image,string filename)
{
	Mat newimage;
	int org_width = image.cols;
	int org_height = image.rows;
	float x_scaling = 1.0 * RESIZEDIM / org_width;
	float y_scaling = 1.0 * RESIZEDIM / org_height;
	float scaling = min(x_scaling, y_scaling);
	int new_width = org_width * scaling;
	int new_height = org_height * scaling;
	int paddingleft = (RESIZEDIM - new_width) / 2;
	int paddingright = RESIZEDIM - paddingleft - new_width;
	int paddingtop = (RESIZEDIM - new_height) / 2;
	int paddingbottom = RESIZEDIM - paddingtop - new_height;
	resize(image, newimage, Size(new_width, new_height),0,0,INTER_AREA);//preserve aspect ratio
	copyMakeBorder(newimage, newimage, paddingtop, paddingbottom, paddingleft, paddingright, BORDER_CONSTANT, Scalar(0, 0, 0));//add padding to make 224x224
	UpdateBBoxCoordinates(filename, new_width, new_height, paddingleft, paddingtop);
	return newimage;
}


int main0()
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
		string labelpath = imagepath;
		imagepath += ".bin";//Change Image type to bin
		labelpath += ".txt";
		if (countBBox("trainlabels" + labelpath) > 1)//Drop Images with more than 1 bounding box
			continue;
		img = resizeImageAndUpdateBBox(img,"trainlabels"+labelpath);//Resize Image to 224x224 while preserving aspect ratio and update BBox coordinates
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
		string labelpath = imagepath;
		imagepath += ".bin";//Change Image type to bin
		labelpath += ".txt";
		if (countBBox("vallabels"+labelpath) > 1)//Drop Images with more than 1 bounding box
			continue;
		img = resizeImageAndUpdateBBox(img,"vallabels"+labelpath);//Resize Image to 224x224 while preserving aspect ratio and update BBox Coordinates
		img.convertTo(img, CV_32F, 1.0 / 255);//Normalise pixel values
		SaveAsBinary("normalisedval/" + imagepath, img);//Save Image as binary in folder "/normalisedval"
	}
	return 0;
}