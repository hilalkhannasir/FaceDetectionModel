#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <winsock2.h>
#include <ws2tcpip.h>
#include<random>
using namespace cv;

#pragma comment(lib, "ws2_32.lib")

#define PORT 13535
#define KERNELS 16
#define KERNELSIZE 3
#define POOLSIZE 2
std::vector<std::vector<float>> ReconstructMatrix(std::vector<float> data, int rows, int cols)//Reconstruction from input stream to a 2D matrix
{
	std::vector<std::vector<float>> matrix;
	for (int i = 0; i < rows; i++)
	{
		matrix.push_back(std::vector<float>());
		for (int j = 0; j < cols; j++)
			matrix.at(i).push_back(data.at(i * rows + j));
	}
	return matrix;
}

std::vector<std::vector<float>> initKernels()//He initialization
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> distributer(0.0, std::sqrt(2.0 / (KERNELSIZE^2)));
	std::vector<std::vector<float>> weights;
	for (int i = 0; i < KERNELSIZE; i++)
	{
		weights.push_back(std::vector<float>());
		for (int j = 0; j < KERNELSIZE; j++)
			weights.at(i).push_back(distributer(gen));
	}
	return weights;
}

float dotproduct(const std::vector<std::vector<float>>& image, const std::vector<std::vector<float>>& kernel,int row,int col)//Dot Product
{
	float sum = 0;
	for (int i = row; i < row + KERNELSIZE; i++)
	{
		for (int j = col; j < col +KERNELSIZE; j++)
			sum += image[i][j] * kernel[i - row][j - col];
	}
	return sum;
}
float ActivationFunction(float value)//ReLU
{
	if (value < 0)
		return 0;
	return value;
}
float Maxwindow(const std::vector<std::vector<float>>& matrix,int row,int col)//Function to find max value in POOLSIZExPOOLSIZE window
{
	float max = -999999;
	for (int i = row; i < POOLSIZE + row; i++)
	{
		for (int j = col; j < POOLSIZE + col; j++)
		{
			if (matrix[i][j] > max)
				max = matrix[i][j];
		}
	}
	return max;	
}
std::vector<std::vector<std::vector<float>>> MaxPooling(const std::vector<std::vector<std::vector<float>>>& ConvolvedImages)
{
	std::vector<std::vector<std::vector<float>>> MaxPooledImages;

	for (int i = 0; i < ConvolvedImages.size(); i++)
	{
		MaxPooledImages.push_back(std::vector<std::vector<float>>());
		for (int j = 0; j <= ConvolvedImages[i].size() - POOLSIZE; j+=POOLSIZE)
		{
			MaxPooledImages[i].push_back(std::vector<float>());
			for (int k = 0; k <= ConvolvedImages[i][j].size() - POOLSIZE; k += POOLSIZE)
			{
				MaxPooledImages[MaxPooledImages.size() - 1][j / 2].push_back(1.0 * Maxwindow(ConvolvedImages[i], j, k));
			}
		}
	}
	return MaxPooledImages;
}

std::vector<std::vector<std::vector<float>>> Convolution(const std::vector<std::vector<std::vector<float>>>& matrices,
	const std::vector<std::vector<std::vector<float>>>& kernels,const std::vector<float>& biases)
{
	std::vector<std::vector<std::vector<float>>> Convolvedimages;
	for (int i = 0; i < matrices.size(); i++)//At each image
	{
		for (int l = 0; l < kernels.size(); l++)//Apply all kernels one by one
		{
			Convolvedimages.push_back(std::vector<std::vector<float>>());
			for (int j = 0; j <= matrices.at(i).size() - KERNELSIZE; j++)//At each row
			{
				Convolvedimages[Convolvedimages.size() - 1].push_back(std::vector<float>());
				for (int k = 0; k <= matrices.at(i).at(j).size() - KERNELSIZE; k++)//and each column
				{
					float value = dotproduct(matrices[i], kernels[l], j, k);
					value += biases[l];//Add Bias value
					Convolvedimages[Convolvedimages.size() - 1][j].push_back(ActivationFunction(value));//Apply Activation Function
				}
			}
		}
	}
	return Convolvedimages;
}



int main()
{
	WSADATA wsaData;
	WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
		std::cerr << "WSAStartup failed" << std::endl;
		return 1;
	}
	else
		std::cout << "---------Started SuccessFully-------------" << std::endl;

	SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
	if (sock == INVALID_SOCKET) {
		std::cerr << "Socket creation failed" << std::endl;
		WSACleanup();
		return 1;
	}
	else
		std::cout << "-------------Socket Creation Successfull------------" << std::endl;
	sockaddr_in client;
	client.sin_family = AF_INET;
	inet_pton(AF_INET, "127.0.0.1", &client.sin_addr.s_addr);
	client.sin_port = htons(PORT);
	if (connect(sock, (sockaddr*)&client, sizeof(client)) == SOCKET_ERROR) {
		std::cerr << "Connection failed" << std::endl;
		closesocket(sock);
		WSACleanup();
		return 1;
	}
	else
		std::cout << "---------------Connection Established-------------" << std::endl;
	int numoffeaturemaps, rows, cols;
	recv(sock, reinterpret_cast<char*>(&numoffeaturemaps), sizeof(numoffeaturemaps), 0);
	recv(sock, reinterpret_cast<char*>(&rows), sizeof(rows), 0);
	recv(sock, reinterpret_cast<char*>(&cols), sizeof(cols), 0);
	numoffeaturemaps = ntohl(numoffeaturemaps);
	rows = ntohl(rows);
	cols = ntohl(cols);
	std::vector<std::vector<float>> featuremaps;
	for (int i = 0; i < numoffeaturemaps; i++)//Recieving batch
	{
		std::vector<float> data(rows*cols*sizeof(float));
		if (recv(sock, reinterpret_cast<char*>(data.data()), rows * cols * sizeof(float), 0) == -1)
		{
			std::cout << "Error Recieving" << std::endl;
		}
		else
		{
			std::cout << "Image Recieved" << std::endl;
			featuremaps.push_back(data);
		}
	}
	std::vector < std::vector<std::vector<float>>> matrices;//Vector to store all the input image matrices
	std::vector<std::vector<std::vector<float>>> kernels;//Vector to store all the kernels
	std::vector<float> biases;//Vector to store all the biases
	for (int i = 0; i < numoffeaturemaps; i++)//Construction 2D vectors, initialising kernel weights and biases
	{
		matrices.push_back(ReconstructMatrix(featuremaps[i], rows, cols));
		kernels.push_back(initKernels());
		biases.push_back(0);
	}
	matrices = Convolution(matrices, kernels,biases);
	matrices = MaxPooling(matrices);//This output is the final product of this layer
	closesocket(sock);
	WSACleanup();
	return 0;
}