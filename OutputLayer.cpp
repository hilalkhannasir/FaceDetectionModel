#include <iostream>
#include<vector>
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib, "ws2_32.lib")

#define PREVPORT 13539

std::vector<float> flatten(std::vector<std::vector<float>> matrix)
{
	std::vector<float> result;
	for (int i = 0; i < matrix.size(); i++)
	{
		for (int j = 1; j < matrix[0].size(); j++)
			result.push_back(matrix[i][j]);
	}
	return result;
}

std::vector<float> MSEDeriv(std::vector<float> y, std::vector<float> y_pred)
{
	std::vector<float> derivvals;
	for (int i = 0; i < y.size(); i++)
	{
		derivvals.push_back(-2.0 * (y[i] - y_pred[i]) / y.size());
	}
	return derivvals;
}

float MSE(std::vector<float> vec1, std::vector<float> vec2)
{
	float sum = 0;
	for (int i = 0; i < vec1.size(); i++)
		sum += pow((vec1[i] - vec2[i]), 2);
	return sum / vec1.size();
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

	SOCKET prevsock = socket(AF_INET, SOCK_STREAM, 0);

	if (prevsock == INVALID_SOCKET) {
		std::cerr << "Previous Layer Socket Creation failed" << std::endl;
		WSACleanup();
		return 1;
	}
	else
		std::cout << "-------------Previous Layer Socket Creation Successfull------------" << std::endl;

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
	int batchsize,numofBBox;
	std::vector<std::vector<float>> BBoxCoords;
	recv(prevsock, reinterpret_cast<char*>(&batchsize), sizeof(batchsize), 0);//Recieving Batch Size
	std::vector<float> MSEs;
	char temp;
	int count = 0;
	float batchloss = 0;
	while (recv(prevsock, &temp, 1, MSG_PEEK) != 0)
	{
		count++;
	/*for (int i = 0; i < batchsize; i++)
	{*/
		recv(prevsock, reinterpret_cast<char*>(&numofBBox), sizeof(numofBBox), 0);//Recieving number of BBox in this image
		for (int j = 0; j < numofBBox; j++)
		{
			BBoxCoords.push_back(std::vector<float>(5));
			recv(prevsock, reinterpret_cast<char*>(BBoxCoords[j].data()), BBoxCoords[j].size() * sizeof(float), 0);//Recieving True Output
		}
		std::vector<float> actualoutput = flatten(BBoxCoords);
		std::vector<float> predictedoutput(numofBBox * 4);
		recv(prevsock, reinterpret_cast<char*>(predictedoutput.data()), predictedoutput.size() * sizeof(float), 0);//Recieving Predicted Output
		MSEs.push_back(MSE(actualoutput, predictedoutput));//Calculating MSE for this Sample
		BBoxCoords.clear();
	//}
		std::cout << "Actual Outputs: ";
		for (int i = 0; i < 4; i++)
			std::cout << actualoutput[i] << ' ';
		std::cout << std::endl;
		std::cout << "Predicted Outputs: ";
		for (int i = 0; i < 4; i++)
			std::cout << predictedoutput[i] << ' ';
		std::cout << std::endl;

		float loss = 0;
		for (int i = 0; i < MSEs.size(); i++)
			loss += MSEs[i];
		loss /= MSEs.size();
		batchloss += loss;
		if (count % 20 == 0)
			std::cout << "Batch Loss: "<< batchloss/20 << std::endl,batchloss = 0;
		std::cout << "Loss: " << loss << std::endl;
		std::vector<float> gradient = MSEDeriv(actualoutput,predictedoutput);
		send(prevsock, (char*)gradient.data(), gradient.size() * sizeof(float), 0);
		MSEs.clear();
	}
	closesocket(prevsock);
	WSACleanup();
	return 0;
}