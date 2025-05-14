#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include<fstream>
#include<random>

#pragma comment(lib, "ws2_32.lib")

#define PREVPORT 13540
#define NEXTPORT 13541
#define OUTNEURON 64
#define LR 0.0001
#define ALPHA 0
std::vector<std::vector<float>> initWeights(int inputsize)//He initialization
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> distributer(0.0, std::sqrt(2.0 / inputsize));
	std::vector<std::vector<float>> weights;
	for (int i = 0; i < inputsize; i++)
	{
		weights.push_back(std::vector<float>());
		for (int j = 0; j < OUTNEURON; j++)
			weights.at(i).push_back(distributer(gen));
	}
	return weights;
}
std::vector<std::vector<float>> LoadWeights(int inputsize, int outputsize)
{
	std::ifstream reader("E:\\weightsandbiases\\WeightsFC3.txt");
	float temp;
	std::vector<std::vector<float>> weights;
	for (int i = 0; i < inputsize; i++)
	{
		weights.push_back(std::vector<float>());
		for (int j = 0; j < outputsize; j++)
		{
			reader >> temp;
			weights[i].push_back(temp);
		}
	}
	reader.close();
	return weights;
}
std::vector<float> LoadBias()
{
	std::ifstream reader("E:\\weightsandbiases\\BiasFC3.txt");
	float temp;
	std::vector<float> bias;
	for (int i = 0; i < OUTNEURON; i++)
	{
		reader >> temp;
		bias.push_back(temp);
	}
	reader.close();
	return bias;
}
void StoreWeights(const std::vector<std::vector<float>>& weights)
{
	std::ofstream writer("WeightsFC3.txt");
	for (int i = 0; i < weights.size(); i++)
	{
		for (int j = 0; j < weights[i].size(); j++)
		{
			writer << weights[i][j] << ' ';
			writer << "\n";
		}
	}
	writer.close();
}

void StoreBias(const std::vector<float>& bias)
{
	std::ofstream writer("BiasFC3.txt");
	for (int i = 0; i < bias.size(); i++)
		writer << bias[i] << ' ';
	writer.close();
}

void UpdateWeights(std::vector<std::vector<float>>& weights, float Lr, const std::vector<std::vector<float>>& deltaw, int batchsize)
{
	for (int i = 0; i < weights.size(); i++)
	{
		for (int j = 0; j < weights[i].size(); j++)
		{
			weights[i][j] -= Lr * deltaw[i][j] / batchsize;
		}
	}
}
std::vector<std::vector<float>> _2DMatrixMul(const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2)
{
	std::vector<std::vector<float>> resultant;
	for (int i = 0; i < mat1[0].size(); i++)
	{
		resultant.push_back(std::vector<float>());
		for (int k = 0; k < mat2[0].size(); k++)
		{
			float sum = 0.0;
			for (int j = 0; j < mat1.size(); j++)
				sum += mat1[j][i] * mat2[j][k];
			resultant[i].push_back(sum);
		}
	}
	return resultant;
}
void UpdateBias(std::vector<float>& bias, float Lr, const std::vector<float>& gradient)
{
	for (int i = 0; i < bias.size(); i++)
		bias[i] -= Lr * gradient[i];
}


std::vector<float> MatrixMul(const std::vector<float>& matrix1, const std::vector < std::vector<float>>& matrix2)
{
	if (matrix1.size() != matrix2.size())
		return std::vector<float>();
	std::vector<float> result;
	for (int j = 0; j < matrix2[0].size(); j++)
	{
		float sum = 0.0;
		for (int i = 0; i < matrix1.size(); i++)
			sum += matrix1[i] * matrix2[i][j];
		result.push_back(sum);
	}
	return result;
}

std::vector<float> BackMul(const std::vector<float>& gradient, const std::vector<std::vector<float>>& weights)
{
	std::vector<float> resultant;
	for (int i = 0; i < weights.size(); i++)
	{
		float sum = 0.0;
		for (int j = 0; j < gradient.size(); j++)
		{
			sum += gradient[j] * weights[i][j];
		}
		resultant.push_back(sum);
	}
	return resultant;
}

std::vector<std::vector<float>> LastMul(const std::vector<float>& mat1, const std::vector<float>& mat2)
{
	std::vector<std::vector<float>> resultant;
	for (int i = 0; i < mat1.size(); i++)
	{
		resultant.push_back(std::vector<float>());
		for (int j = 0; j < mat2.size(); j++)
			resultant[i].push_back(mat1[i] * mat2[j]);
	}
	return resultant;
}
float maxval(const std::vector<float>& data)
{
	float maximum = data[0];
	for (int i = 0; i < data.size(); i++)
		maximum = max(data[i], maximum);
	return maximum;
}
std::vector<float> Normalise(std::vector<float>& data)
{
	float maximum = maxval(data);
	for (int i = 0; i < data.size(); i++)
		data[i] /= maximum;
	return data;
}
float ActivationFunction(float value)//ReLU
{
	if (value < 0)
		return value*ALPHA;
	return value;
}
std::vector<float> DerivReLU(const std::vector<float>& predicted)
{
	std::vector<float> derivvals;
	for (int i = 0; i < predicted.size(); i++)
	{
		if (predicted[i] > 0)
			derivvals.push_back(1);
		else
			derivvals.push_back(ALPHA);
	}
	return derivvals;
}

std::vector<float> ColumnWiseAverage(const std::vector<std::vector<float>>& matrix)
{
	std::vector<float> result(matrix[0].size(), 0);
	for (int i = 0; i < matrix[0].size(); i++)
	{
		for (int j = 0; j < matrix.size(); j++)
		{
			result[i] += matrix[j][i];
		}
		result[i] / matrix.size();
	}
	return result;
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

	SOCKET prevsock = socket(AF_INET, SOCK_STREAM, 0), serversock = socket(AF_INET, SOCK_STREAM, 0), nextsock;

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

	sockaddr_in prevclient, server;
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

	//-------------------------------Receiveing Data-------------------------------------------
	int batchsize, inputsize;
	recv(prevsock, reinterpret_cast<char*>(&batchsize), sizeof(batchsize), 0);
	recv(prevsock, reinterpret_cast<char*>(&inputsize), sizeof(inputsize), 0);

	std::vector<std::vector<float>> fullyconnecteddata, weights, output;
	std::vector<float> biases;//Vector to store all the biases
	int outputsize = OUTNEURON;
	send(nextsock, reinterpret_cast<char*>(&batchsize), sizeof(batchsize), 0);//Sending Batch Size
	send(nextsock, reinterpret_cast<char*>(&outputsize), sizeof(outputsize), 0);//Sending the size of output of this layer
	std::vector<std::vector<float>> BBoxCoords;
	char temp;
	int numofBBox;
	/*weights = initWeights(inputsize);
	for (int i = 0; i < OUTNEURON; i++)
		biases.push_back(0);*/
	weights = LoadWeights(inputsize, OUTNEURON);
	biases = LoadBias();
	while (recv(prevsock, &temp, 1, MSG_PEEK) != 0)
	{
		for (int i = 0; i < batchsize; i++)
		{
			std::vector<float> imagedata(inputsize);

			recv(prevsock, reinterpret_cast<char*>(&numofBBox), sizeof(numofBBox), 0);//Recieving number of BBox in this image
			std::cout << numofBBox << " BBoxes Recieved" << std::endl;

			for (int j = 0; j < numofBBox; j++)
			{
				BBoxCoords.push_back(std::vector<float>(5));
				recv(prevsock, reinterpret_cast<char*>(BBoxCoords[j].data()), BBoxCoords[j].size() * sizeof(float), 0);//Recieving BBox Coords
			}
			recv(prevsock, reinterpret_cast<char*>(imagedata.data()), imagedata.size() * sizeof(float), 0);//Recieving Image Data
			//imagedata = Normalise(imagedata);
			std::vector<float> tempoutput = MatrixMul(imagedata, weights);// mx
			for (int j = 0; j < OUTNEURON; j++)// + c
				tempoutput[j] += biases[j];
			for (int j = 0; j < tempoutput.size(); j++)
			{
				tempoutput[j] = ActivationFunction(tempoutput[j]);//ReLU Function
			}
			output.push_back(tempoutput);
			numofBBox = BBoxCoords.size();
			send(nextsock, reinterpret_cast<char*>(&numofBBox), sizeof(numofBBox), 0);//Sending number of BBoxes in the image
			for (int j = 0; j < numofBBox; j++)
				send(nextsock, reinterpret_cast<char*>(BBoxCoords[j].data()), BBoxCoords[j].size() * sizeof(float), 0);//Sending BBox Coordinates

			send(nextsock, (char*)tempoutput.data(), tempoutput.size() * sizeof(float), 0);//Sending data
			fullyconnecteddata.push_back(imagedata);
			BBoxCoords.clear();
		}
		//-----------------------------Back Propagation-------------------------------------
		std::vector<std::vector<float>> nextlayergradient(batchsize, std::vector<float>(OUTNEURON, 0));//Vector of size (batchsizexOUTNEURON)
		for (int i = 0; i < nextlayergradient.size(); i++)
			recv(nextsock, (char*)nextlayergradient[i].data(), nextlayergradient[i].size() * sizeof(float), 0);//Receiveing Gradients
		std::vector<std::vector<float>> gradient;
		for (int i = 0; i < nextlayergradient.size(); i++)
		{
			std::vector<float> derivReLU = DerivReLU(output[i]);//Calculating Derivative of Sigmoid
			gradient.push_back(std::vector<float>());
			for (int j = 0; j < nextlayergradient[i].size(); j++)
			{
				gradient[i].push_back(derivReLU[j] * nextlayergradient[i][j]);
			}
		}
		std::vector<float> backgradient, tempgradient;

		std::vector<std::vector<float>> deltaw = _2DMatrixMul(fullyconnecteddata, gradient);//Matrix Multiplication (nx1 * 1xOUTNEURON = n x OUTNEURON)
		for (int i = 0; i < gradient.size(); i++)
		{
			tempgradient = BackMul(gradient[i], weights);
			backgradient.insert(backgradient.end(), tempgradient.begin(), tempgradient.end());
		}
		UpdateWeights(weights, LR, deltaw, batchsize);
		UpdateBias(biases, LR, ColumnWiseAverage(gradient));
		StoreWeights(weights);
		StoreBias(biases);
		send(prevsock, (char*)backgradient.data(), backgradient.size() * sizeof(float), 0);//Back Propagation
		std::cout << "Back Propagating..." << std::endl;
		fullyconnecteddata.clear();
	}

	closesocket(prevsock);
	WSACleanup();
	return 0;
}