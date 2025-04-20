#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <winsock2.h>
#include <ws2tcpip.h>
#include<random>
using namespace cv;

#pragma comment(lib, "ws2_32.lib")

#define PREVPORT 13535
#define NEXTPORT 13536
#define KERNELS 8
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

std::vector<float> FlattenMatrix(std::vector<std::vector<float>> matrix)
{
	std::vector<float> flattened;
	for (int i = 0; i < matrix.size(); i++)
	{
		for (int j = 0; j < matrix[0].size(); j++)
			flattened.push_back(matrix[i][j]);
	}
	return flattened;
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

std::vector<std::vector<std::vector<float>>> Convolution(const std::vector<std::vector<float>>& image,
	const std::vector<std::vector<std::vector<float>>>& kernels,const std::vector<float>& biases)
{
	std::vector<std::vector<std::vector<float>>> Convolvedimages;
	for (int l = 0; l < kernels.size(); l++)//Apply all kernels one by one
	{
		Convolvedimages.push_back(std::vector<std::vector<float>>());
		for (int j = 0; j <= image.size() - KERNELSIZE; j++)//At each row
		{
			Convolvedimages[Convolvedimages.size() - 1].push_back(std::vector<float>());
			for (int k = 0; k <= image.at(j).size() - KERNELSIZE; k++)//and each column
			{
				float value = dotproduct(image, kernels[l], j, k);
				value += biases[l];//Add Bias value
				Convolvedimages[Convolvedimages.size() - 1][j].push_back(ActivationFunction(value));//Apply Activation Function
			}
		}
	}
	return Convolvedimages;
}



int main()
{													//-------------------------Establishing Connections-------------------------------
	WSADATA wsaData;
	WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
		std::cerr << "WSAStartup failed" << std::endl;
		return 1;
	}
	else
		std::cout << "---------Started SuccessFully-------------" << std::endl;

	SOCKET prevsock = socket(AF_INET, SOCK_STREAM, 0),serversock = socket(AF_INET,SOCK_STREAM,0),nextsock;

	if (prevsock == INVALID_SOCKET) {
		std::cerr << "Previous Layer Socket Creation failed" << std::endl;
		WSACleanup();
		return 1;
	}
	else
		std::cout << "-------------Previous Layer Socket Creation Successfull------------" << std::endl;
	if (serversock == INVALID_SOCKET) {
		std::cerr << "Next Layer Socket Creation failed" << std::endl;
		WSACleanup();
		return 1;
	}
	else
		std::cout << "-------------Next Layer Socket Creation Successfull------------" << std::endl;

	sockaddr_in prevclient,server;
	prevclient.sin_family = AF_INET;
	inet_pton(AF_INET, "127.0.0.1", &prevclient.sin_addr.s_addr);
	prevclient.sin_port = htons(PREVPORT);
	if (connect(prevsock, (sockaddr*)&prevclient, sizeof(prevclient)) == SOCKET_ERROR) {
		std::cerr << "Previous Layer Socket Connection failed" << std::endl;
		closesocket(prevsock);
		WSACleanup();
		return 1;
	}
	else
		std::cout << "---------------Previous Layer Socket Connection Established-------------" << std::endl;

	server.sin_family = AF_INET;
	inet_pton(AF_INET, "127.0.0.1", &server.sin_addr.s_addr);
	server.sin_port = htons(NEXTPORT);
	if (bind(serversock, (sockaddr*)&server, sizeof(server)) < 0)
	{
		std::cout << "Error at Binding" << std::endl;
	}
	else
		std::cout << "------------Binding Successfull-----------" << std::endl;
	if (listen(serversock, 1) == SOCKET_ERROR)
	{
		perror("Listen failed");
		exit(EXIT_FAILURE);
	}
	else
		std::cout << "Listening..." << std::endl;

	nextsock = accept(serversock, NULL, NULL);
	if (nextsock == INVALID_SOCKET)
	{
		std::cout << "Error Accepting" << std::endl;
		WSACleanup();
		return 0;
	}
	else
		std::cout << "-------Connection Accepted---------" << std::endl;

														//-------------------------------Connections Establised-------------------------------------
	
	int batchsize, rows, cols,numofBBox;

	recv(prevsock, reinterpret_cast<char*>(&batchsize), sizeof(batchsize), 0);
	recv(prevsock, reinterpret_cast<char*>(&rows), sizeof(rows), 0);
	recv(prevsock, reinterpret_cast<char*>(&cols), sizeof(cols), 0);
	std::vector<std::vector<std::vector<float>>> kernels,matrices;//Vector to store all the kernels
	std::vector<float> biases;//Vector to store all the biases
	for (int i = 0; i < KERNELS; i++)//Construction 2D vectors, initialising kernel weights and biases
	{
		kernels.push_back(initKernels());
		biases.push_back(0);
	}
	int numofimages = KERNELS;
	int newrows = (rows - KERNELSIZE + 1)/POOLSIZE;
	int newcols = newrows;
	int numofkernels = KERNELS;
	send(nextsock, reinterpret_cast<char*>(&batchsize), sizeof(batchsize), 0);
	send(nextsock, reinterpret_cast<char*>(&numofimages), sizeof(numofimages), 0);
	send(nextsock, reinterpret_cast<char*>(&newrows), sizeof(newrows), 0);
	send(nextsock, reinterpret_cast<char*>(&newcols), sizeof(newcols), 0);
	send(nextsock, reinterpret_cast<char*>(&numofkernels), sizeof(numofkernels), 0);
	std::vector<std::vector<float>> image,BBoxCoords;
	std::vector<float>data(rows*cols);
	char temp;
	while (recv(prevsock,&temp,1,MSG_PEEK) != 0)
	{
		for (int i = 0; i < batchsize; i++)
		{
			recv(prevsock, reinterpret_cast<char*>(&numofBBox), sizeof(numofBBox), 0);//Recieving number of BBox in this image
			for (int j = 0; j < numofBBox; j++)
			{
				BBoxCoords.push_back(std::vector<float>(5));
				recv(prevsock, reinterpret_cast<char*>(BBoxCoords[j].data()), BBoxCoords[j].size()*sizeof(float), 0);//Recieving BBox Coords
			}
			recv(prevsock, reinterpret_cast<char*>(data.data()), rows * cols * sizeof(float), 0);//Recieving Image Data
			image = ReconstructMatrix(data, rows, cols);
			matrices = Convolution(image, kernels, biases);
			matrices = MaxPooling(matrices);
			int numofBBox = BBoxCoords.size();
			
			send(nextsock, reinterpret_cast<char*>(&numofBBox), sizeof(numofBBox), 0);//Sending number of BBoxes in the image
			for (int j = 0; j < numofBBox; j++)
				send(nextsock, reinterpret_cast<char*>(BBoxCoords[j].data()), BBoxCoords[j].size() * sizeof(float), 0);//Sending BBox Coordinates

			for (int j = 0; j < numofimages; j++)
			{
				std::vector<float>image = FlattenMatrix(matrices[j]);
				if (send(nextsock, (char*)image.data(), image.size() * sizeof(float), 0) == -1)//Sending Image Data
					std::cout << "Error Sending Image" << std::endl;
				/*else
					std::cout << "Image" << j << " Sent" << std::endl;*/

			}
			BBoxCoords.clear();
		}
		float error;
		recv(nextsock, reinterpret_cast<char*>(&error), sizeof(error), 0);
		//Weight Adjustment
		std::cout << "Recieving Feedback" << std::endl;
		send(prevsock, reinterpret_cast<char*>(&error), sizeof(error), 0);
	}
	closesocket(nextsock);
	closesocket(prevsock);
	WSACleanup();
	return 0;
}