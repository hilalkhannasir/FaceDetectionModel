#include <iostream>
#include <filesystem>
#include<vector>
#include<fstream>
#include <winsock2.h>
#include <ws2tcpip.h>
namespace fs = std::filesystem;
#pragma comment(lib, "ws2_32.lib")
#define PORT 13535
#define BATCHSIZE 8
#define ROWS 224
#define COLS 224

std::vector<char> LoadBinaryData(const std::string& filename)
{
	std::ifstream binfile(filename, std::ios::binary | std::ios::ate);
	if (!binfile.is_open())
	{
		std::cout << "Error Opening file" << std::endl;
		return std::vector<char>();
	}
	std::streamsize filesize=  binfile.tellg();
	binfile.seekg(0, std::ios::beg);
	std::vector<char> filedata(filesize);
	if (binfile.read(filedata.data(), filesize))
	{
		binfile.close();
		return filedata;
	}
	else
	{
		std::cout << "Error Loading Data" << std::endl;
		return std::vector<char>();
	}

}
std::vector<std::vector<float>> ReadBBoxCoord(const std::string& filename)
{
	std::vector<std::vector<float>> BBoxCoords;
	std::ifstream reader(filename);
	float temp;
	while (reader >> temp)
	{
		BBoxCoords.push_back(std::vector<float>());
		BBoxCoords.at(BBoxCoords.size() - 1).push_back(temp);
		for (int i = 0; i < 4; i++)
		{
			reader >> temp;
			BBoxCoords.at(BBoxCoords.size() - 1).push_back(temp);
		}
	}
	reader.close();
	return BBoxCoords;
}

std::string splitstring(const std::string& str, char split)
{
	for (int i = 0; i < str.size(); i++)
	{
		if (str[i] == split)
			return str.substr(i+1, str.size());
	}
	return str;
}

int main()
{
	WSADATA wsaData;
	SOCKET serverSocket, clientSocket;
	sockaddr_in serverAddr;
	fs::directory_iterator Start{ "normalisedtrain" };
	fs::directory_iterator End{};
	std::vector<std::vector<float>> BBox;
	auto Iter(Start);
	std::vector<char> batchelement;
	std::string path,labelpath;
	int batchsize = BATCHSIZE;
	int rows = ROWS;
	int cols = COLS;
	int numofBBox,counter;
	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
	{
		std::cout << "Error at Startup" << std::endl;
		return 0;
	}
	else
		std::cout << "---------Started SuccessFully-------------" << std::endl;

	serverSocket = socket(AF_INET, SOCK_STREAM, 0);
	if (serverSocket == INVALID_SOCKET)
	{
		std::cout << "Error at ListenSocket" << std::endl;
		WSACleanup();
		return 0;
	}
	else
		std::cout << "-------------Socket Creation Successfull------------" << std::endl;
	serverAddr.sin_family = AF_INET;
	inet_pton(serverAddr.sin_family, "127.0.0.1", &serverAddr.sin_addr.s_addr);
	serverAddr.sin_port = htons(PORT);

	if (bind(serverSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) < 0)
	{
		std::cout << "Error at Binding" << std::endl;
	}
	else
		std::cout << "------------Binding Successfull-----------" << std::endl;
	if (listen(serverSocket, 1) == SOCKET_ERROR)
	{
		perror("Listen failed");
		exit(EXIT_FAILURE);
	}
	else
		std::cout << "Listening..." << std::endl;

	clientSocket = accept(serverSocket, NULL, NULL);
	if (clientSocket == INVALID_SOCKET)
	{
		std::cout << "Error Accepting" << std::endl;
		WSACleanup();
		return 0;
	}
	else
		std::cout << "-------Connection Accepted---------" << std::endl;
	
	send(clientSocket, reinterpret_cast<char*>(&batchsize), sizeof(batchsize), 0);
	send(clientSocket, reinterpret_cast<char*>(&rows), sizeof(rows), 0);
	send(clientSocket, reinterpret_cast<char*>(&cols), sizeof(cols), 0);
	int epochs = 10;
	for (int epoch = 0; epoch < epochs; epoch++)
	{
		fs::directory_iterator Start{ "normalisedtrain" };
		fs::directory_iterator End{};
		auto Iter(Start);
		for (int k = 0; k < 1000; k+=batchsize, Iter++)
		{
			counter = 0;
			for (int i = k; i < k + BATCHSIZE && Iter != End; i++, Iter++)
			{
				counter++;
				path = Iter->path().string();
				batchelement = LoadBinaryData(path);
				if (batchelement.size() > 0)
				{
					labelpath = splitstring(path, '\\');
					labelpath = labelpath.substr(0, labelpath.size() - 4) + ".txt";
					BBox = ReadBBoxCoord("normalisedtrainlabels/" + labelpath);
					numofBBox = BBox.size();
					send(clientSocket, reinterpret_cast<char*>(&numofBBox), sizeof(numofBBox), 0);//Sending number of BBoxes in the image
					std::cout << numofBBox << " BBoxes sent" << std::endl;
					for (int j = 0; j < numofBBox; j++)
						send(clientSocket, reinterpret_cast<char*>(BBox[j].data()), BBox[j].size() * sizeof(float), 0);//Sending BBox Coordinates
					if (send(clientSocket, (char*)batchelement.data(), batchelement.size(), 0) == -1)//Sending Image Data
						std::cout << "Error Sending" << std::endl;
					else
						std::cout << "Image" << i << " Sent" << std::endl;
				}
				else
				{
					std::cout << "Error in Loading Data. Exiting Program" << std::endl;
					return 0;
				}
			}
			float error;
			recv(clientSocket, reinterpret_cast<char*>(&error), sizeof(error), 0);
			std::cout << "Recieving Feedback" << std::endl;
		}
		
	}
	closesocket(clientSocket);
	closesocket(serverSocket);
	WSACleanup();
	return 0;
}
