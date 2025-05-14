#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <winsock2.h>
#include <ws2tcpip.h>
#include<fstream>
#include<random>
using namespace cv;

#pragma comment(lib, "ws2_32.lib")

#define PREVPORT 13535
#define NEXTPORT 13536
#define KERNELS 8
#define KERNELSIZE 3
#define POOLSIZE 2
#define LR 0.001
#define ALPHA 0
std::vector<std::vector<float>> ReconstructMatrix(const std::vector<float>& data, int rows, int cols)//Reconstruction from input stream to a 2D matrix
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
float Sum(const std::vector<std::vector<float>>& matrix)
{
	float sum = 0.0;
	for (int i = 0; i < matrix.size(); i++)
	{
		for (int j = 0; j < matrix[i].size(); j++)
			sum += matrix[i][j];
	}
	return sum;
}

void UpdateBias(std::vector<float>& bias, const std::vector < std::vector<std::vector<float>>>& gradients,int batchsize)
{
	std::vector<float> temp(KERNELS, 0);
	int count = 0;
	while (count < gradients.size())//Calculating Average of all outputs where bias contributed
	{
		for (int i = count; i < count + KERNELS; i++)
		{
			temp[i - count] += Sum(gradients[i]);
		}
		count += KERNELS;
	}
	for (int i = 0; i < temp.size(); i++)
	{
		std::cout << "Updating Bias by " << LR * temp[i] / batchsize << std::endl;
		bias[i] -= LR * temp[i]/batchsize;
	}
}

std::vector<std::vector<std::vector<float>>> LoadWeights()
{
	std::ifstream reader("E:\\weightsandbiases\\Weights1.txt");
	float temp;
	std::vector<std::vector<std::vector<float>>> weights;
	for (int k = 0; k < KERNELS; k++)
	{
		weights.push_back(std::vector<std::vector<float>>());
		for (int i = 0; i < KERNELSIZE; i++)
		{
			weights[k].push_back(std::vector<float>());
			for (int j = 0; j < KERNELSIZE; j++)
			{
				reader >> temp;
				weights[k][i].push_back(temp);
			}
		}
	}
	reader.close();
	return weights;
}
std::vector<float> LoadBias()
{
	std::ifstream reader("E:\\weightsandbiases\\Bias1.txt");
	float temp;
	std::vector<float> bias;
	for (int i = 0; i < KERNELS; i++)
	{
		reader >> temp;
		bias.push_back(temp);
	}
	reader.close();
	return bias;
}



void StoreWeights(const std::vector<std::vector<std::vector<float>>>& weights)
{
	std::ofstream writer("Weights1.txt");
	for (int i = 0; i < weights.size(); i++)
	{
		for (int j = 0; j < weights[i].size(); j++)
		{
			for (int k = 0; k < weights[i][j].size(); k++)
			{
				writer << weights[i][j][k] << ' ';
			}
			writer << "\n";
		}
	}
	writer.close();
}

void StoreBias(const std::vector<float>& bias)
{
	std::ofstream writer("Bias1.txt");
	for (int i = 0; i < bias.size(); i++)
		writer << bias[i] << ' ';
	writer.close();
}

std::vector<std::vector<std::vector<float>>> UnPool(const std::vector<std::vector<std::vector<float>>>& gradients,
	const std::vector<std::vector<std::vector<float>>>& pooled, const std::vector<std::vector<std::vector<float>>>& convolved)
{
	std::vector<std::vector<std::vector<float>>> Unpooled;
	Unpooled.resize(convolved.size());
	bool padding = false;
	if (convolved[0].size() % 2 != 0)
		padding = true;
	for (int i = 0; i < convolved.size(); i++)
	{
		Unpooled[i].resize(convolved[i].size());
		for (int j = 0; j <= convolved[i].size() - POOLSIZE; j += POOLSIZE)
		{
			for (int k = 0; k <= convolved[i][j].size() - POOLSIZE; k += POOLSIZE)
			{
				for (int l = j; l < j + POOLSIZE; l++)
				{
					for (int m = k; m < k + POOLSIZE; m++)
					{
						if (pooled[i][j / 2][k / 2] != convolved[i][l][m] or pooled[i][j / 2][k / 2] <= 0)//Derivative of ReLU
							Unpooled[i][l].push_back(ALPHA);
						else
							Unpooled[i][l].push_back(gradients[i][j / 2][k / 2]);
					}
				}

			}
		}
	}
	if (padding)
	{
		for (int i = 0; i < Unpooled.size(); i++)
		{
			for (int j = 0; j < Unpooled[i].size(); j++)
			{
				Unpooled[i][j].push_back(0);
			}
			Unpooled[i][Unpooled[i].size() - 1].resize(Unpooled[i].size(), 0);
		}
	}

	return Unpooled;
}



std::vector<float> FlattenMatrix(const std::vector<std::vector<float>>& matrix)
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
	std::normal_distribution<float> distributer(0.0, std::sqrt(2.0 / (KERNELSIZE*KERNELSIZE)));
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
		return value*ALPHA;
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
//
//std::vector<std::vector<std::vector<float>>> Convolution(const std::vector<std::vector<float>>& image,
//	const std::vector<std::vector<std::vector<float>>>& kernels,const std::vector<float>& biases)
//{
//	std::vector<std::vector<std::vector<float>>> Convolvedimages;
//	for (int l = 0; l < kernels.size(); l++)//Apply all kernels one by one
//	{
//		Convolvedimages.push_back(std::vector<std::vector<float>>());
//		for (int j = 0; j <= image.size() - KERNELSIZE; j++)//At each row
//		{
//			Convolvedimages[Convolvedimages.size() - 1].push_back(std::vector<float>());
//			for (int k = 0; k <= image.at(j).size() - KERNELSIZE; k++)//and each column
//			{
//				float value = dotproduct(image, kernels[l], j, k);
//				value += biases[l];//Add Bias value
//				Convolvedimages[Convolvedimages.size() - 1][j].push_back(ActivationFunction(value));//Apply Activation Function
//			}
//		}
//	}
//	return Convolvedimages;
//}


std::vector<std::vector<float>> Convolve(const std::vector<std::vector<float>>& matrix, const std::vector<std::vector<float>>& kernel)
{
	std::vector<std::vector<float>> convolved;
	for (int i = 0; i <= matrix.size() - kernel.size(); i++)
	{
		convolved.push_back(std::vector<float>());
		for (int j = 0; j <= matrix[i].size() - kernel.size(); j++)
		{
			convolved[convolved.size() - 1].push_back(dotproduct(matrix, kernel, i, j));
		}
	}
	return convolved;
}
std::vector<std::vector<float>> AddMatrices(const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2)
{
	std::vector<std::vector<float>> result;
	for (int i = 0; i < mat1.size(); i++)
	{
		result.push_back(std::vector<float>());
		for (int j = 0; j < mat1[i].size(); j++)
			result[i].push_back(mat1[i][j] + mat2[i][j]);
	}
	return result;
}

void UpdateWeights(std::vector<std::vector<std::vector<float>>>& kernels, const std::vector<std::vector<std::vector<float>>>& gradients,
	const std::vector<std::vector<std::vector<float>>>& inputs,int batchsize)
{
	std::vector<std::vector<std::vector<float>>> deltaw(KERNELS, std::vector<std::vector<float>>(KERNELSIZE, std::vector<float>(KERNELSIZE, 0)));
	for (int i = 0; i < inputs.size(); i++)
	{
		for (int j = i * KERNELS; j < (i + 1) * KERNELS; j++)
		{
			deltaw[j - (i * KERNELS)] = AddMatrices(deltaw[j - (i * KERNELS)], Convolve(inputs[i], gradients[j]));
		}
	}
	for (int i = 0; i < deltaw.size(); i++)
	{
		for (int j = 0; j < deltaw[i].size(); j++)
		{
			for (int k = 0; k < deltaw[j].size(); k++)
			{
				deltaw[i][j][k] *= LR/batchsize;
				std::cout << "Updating Weight by " << deltaw[i][j][k] << std::endl;

				kernels[i][j][k] -= deltaw[i][j][k];
			}
		}
	}
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
	std::vector<std::vector<std::vector<float>>> kernels, convolved,input, pooled;//Vector to store all the kernels,input,convolvedimages and pooled images
	std::vector<float> biases;//Vector to store all the biases
	for (int i = 0; i < KERNELS; i++)//Construction 2D vectors, initialising kernel weights and biases
	{
		kernels.push_back(initKernels());
		biases.push_back(0);
	}
	//kernels = LoadWeights();
	//biases = LoadBias();
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
	int newnumofimages = KERNELS*batchsize;
	while (recv(prevsock,&temp,1,MSG_PEEK) != 0)
	{
		for (int i = 0; i < batchsize; i++)
		{
			recv(prevsock, reinterpret_cast<char*>(&numofBBox), sizeof(numofBBox), 0);//Recieving number of BBox in this image
			std::cout << numofBBox << " BBoxes Recieved" << std::endl;

			for (int j = 0; j < numofBBox; j++)
			{
				BBoxCoords.push_back(std::vector<float>(5));
				recv(prevsock, reinterpret_cast<char*>(BBoxCoords[j].data()), BBoxCoords[j].size()*sizeof(float), 0);//Recieving BBox Coords
			}
			recv(prevsock, reinterpret_cast<char*>(data.data()), rows * cols * sizeof(float), 0);//Recieving Image Data

			image = ReconstructMatrix(data, rows, cols);
			input.push_back(image);
			std::vector<std::vector<std::vector<float>>> temp;
			float count = 0.0;
			for (int j = 0; j < KERNELS; j++)
			{
				temp.push_back(Convolve(image, kernels[j]));//applying convolution
				for (int k = 0; k < temp[j].size(); k++)
				{
					for (int l = 0; l < temp[j][k].size(); l++)
					{
						temp[j][k][l] += biases[j];//Adding bias
						temp[j][k][l] = ActivationFunction(temp[j][k][l]);//Applying Activation
						if (temp[j][k][l] == 0)
							count++;
					}
				}
			}
			std::cout << "Percentage 0 Neurons: " << 1.0 * count / (temp.size() * temp[0].size() * temp[0][0].size()) * 100 << std::endl;

			convolved.insert(convolved.end(), temp.begin(), temp.end());
			temp = MaxPooling(temp);
			pooled.insert(pooled.end(), temp.begin(), temp.end());
			numofBBox = BBoxCoords.size();
			
			send(nextsock, reinterpret_cast<char*>(&numofBBox), sizeof(numofBBox), 0);//Sending number of BBoxes in the image
			std::cout << numofBBox << " BBoxes Sent " << std::endl;
			for (int j = 0; j < numofBBox; j++)
				send(nextsock, reinterpret_cast<char*>(BBoxCoords[j].data()), BBoxCoords[j].size() * sizeof(float), 0);//Sending BBox Coordinates

			for (int j = 0; j < numofimages; j++)
			{
				std::vector<float>image = FlattenMatrix(pooled[j+(i*numofimages)]);
				if (send(nextsock, (char*)image.data(), image.size() * sizeof(float), 0) == -1)//Sending Image Data
					std::cout << "Error Sending Image" << std::endl;
			}
			BBoxCoords.clear();
		}
		int outrows, outcols;
		outrows = pooled[0].size();
		outcols = pooled[0][0].size();
		std::vector<float> nextlayergradient(outrows* outcols), gradient;
		std::vector<std::vector<std::vector<float>>> unpooledgradients;
		for (int j = 0; j < newnumofimages; j++)
		{
			if (recv(nextsock, (char*)nextlayergradient.data(), nextlayergradient.size() * sizeof(float), 0) < 0)
				std::cout << "Error Recieving Gradients" << std::endl;//Recieving Gradients
			unpooledgradients.push_back(ReconstructMatrix(nextlayergradient, outrows, outcols));
		}
		unpooledgradients = UnPool(unpooledgradients, pooled, convolved);//Unpooling + ReLU derivative
		UpdateBias(biases, unpooledgradients,batchsize);
		UpdateWeights(kernels, unpooledgradients, input,batchsize);
		StoreWeights(kernels);
		StoreBias(biases);
		input.clear();
		convolved.clear();
		pooled.clear();
		int error = 0;
		send(prevsock, reinterpret_cast<char*>(&error), sizeof(error), 0);
		std::cout << "BackProp completed" << std::endl;
	}
	closesocket(nextsock);
	closesocket(prevsock);
	WSACleanup();
	return 0;
}