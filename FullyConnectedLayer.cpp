#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include<random>

#pragma comment(lib, "ws2_32.lib")

#define PREVPORT 13538
#define NEXTPORT 13539
#define OUTNEURON 4
std::vector<std::vector<float>> initWeights(int numofimages,int rows,int cols,int BBoxes)//He initialization
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-1.0, 1.0); 
	std::vector<std::vector<float>> weights;

	for (int i = 0; i < numofimages*rows*cols; i++)//Input Size
	{
		weights.push_back(std::vector<float>());
		for (int j = 0; j < OUTNEURON*BBoxes; j++)
			weights.at(weights.size() - 1).push_back(dis(gen));
	}
	return weights;
}

std::vector<float> MatrixMul(std::vector<float> matrix1,std::vector < std::vector<float>> matrix2)
{
	if (matrix1.size() != matrix2.size())
		return std::vector<float>();
	std::vector<float> result;
	for (int j = 0; j < matrix2[0].size(); j++)
	{
		int sum = 0;
		for (int i = 0; i < matrix1.size(); i++)
			sum += matrix1[i] * matrix2[i][j];
		result.push_back(sum);
	}
	return result;
}

std::vector<float> ActivationFunction(std::vector<float> values)//Sigmoid
{
	std::vector<float> results;
	for (int i = 0; i < values.size(); i++)
	{
		results.push_back(1.0 / (1.0 + std::exp(-values[i])));
	}
	return results;
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
	int batchsize, numofimages, rows, cols;
	recv(prevsock, reinterpret_cast<char*>(&batchsize), sizeof(batchsize), 0);
	recv(prevsock, reinterpret_cast<char*>(&numofimages), sizeof(numofimages), 0);
	recv(prevsock, reinterpret_cast<char*>(&rows), sizeof(rows), 0);
	recv(prevsock, reinterpret_cast<char*>(&cols), sizeof(cols), 0);
	std::vector<std::vector<float>> weights;
	std::vector<float> fullyconnecteddata;
	std::vector<float> biases,output;//Vector to store all the biases
	
	send(nextsock, reinterpret_cast<char*>(&batchsize), sizeof(batchsize), 0);//Sending Batch Size
	std::vector<std::vector<float>> BBoxCoords;
	std::vector<float>data(rows * cols);
	char temp;
	int numofBBox;
	while (recv(prevsock, &temp, 1, MSG_PEEK) != 0)
	{
		for (int i = 0; i < batchsize; i++)
		{
			biases.clear();
			recv(prevsock, reinterpret_cast<char*>(&numofBBox), sizeof(numofBBox), 0);//Recieving number of BBox in this image
			for (int j = 0; j < numofBBox; j++)
			{
				BBoxCoords.push_back(std::vector<float>(5));
				recv(prevsock, reinterpret_cast<char*>(BBoxCoords[j].data()), BBoxCoords[j].size() * sizeof(float), 0);//Recieving BBox Coords
			}
			weights = initWeights(numofimages, rows, cols,numofBBox);
			for (int i = 0; i < OUTNEURON*numofBBox; i++)
				biases.push_back(0);

			for (int j = 0; j < numofimages; j++)
			{
				recv(prevsock, reinterpret_cast<char*>(data.data()), rows * cols * sizeof(float), 0);//Recieving Image Data
				fullyconnecteddata.insert(fullyconnecteddata.end(), data.begin(), data.end());//Appending at the end of fullyconnected Vector
			}
			std::cout << "Element "<< i << " Recieved" << std::endl;
			output = MatrixMul(fullyconnecteddata, weights);// mx
			for (int j = 0; j < OUTNEURON; j++)// + c
				output[j] += biases[j];
			output = ActivationFunction(output);//Sigmoid Function

			numofBBox = BBoxCoords.size();
			send(nextsock, reinterpret_cast<char*>(&numofBBox), sizeof(numofBBox), 0);//Sending number of BBoxes in the image
			for (int j = 0; j < numofBBox; j++)
				send(nextsock, reinterpret_cast<char*>(BBoxCoords[j].data()), BBoxCoords[j].size() * sizeof(float), 0);//Sending BBox Coordinates

			send(nextsock, (char*)output.data(), output.size() * sizeof(float), 0);//Sending Output prediction
			BBoxCoords.clear();
			fullyconnecteddata.clear();
		}
		float error = 0;
		recv(nextsock, reinterpret_cast<char*>(&error), sizeof(error), 0);
		//Weight Adjustment
		send(prevsock, reinterpret_cast<char*>(&error), sizeof(error), 0);
		std::cout << "Sending Feedback" << std::endl;

	}

	closesocket(prevsock);
	WSACleanup();
	return 0;
}