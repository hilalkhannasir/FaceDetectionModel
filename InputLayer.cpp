#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <filesystem>
#include<vector>
#include<fstream>
#include <winsock2.h>
#include <ws2tcpip.h>
using namespace cv;
namespace fs = std::filesystem;
#pragma comment(lib, "ws2_32.lib")
#define PORT 13535
#define BATCHSIZE 16
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

int main()
{
	WSADATA wsaData;
	SOCKET serverSocket, clientSocket;
	sockaddr_in serverAddr;
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
	fs::directory_iterator Start{ "normalisedtrain" };
	fs::directory_iterator End{};
	auto Iter(Start);
	/*for (; Iter != End; Iter++)
	{*/
		int batchsize = htonl(BATCHSIZE);
		int rows = htonl(ROWS);
		int cols = htonl(COLS);
		send(clientSocket, reinterpret_cast<char*>(&batchsize), sizeof(batchsize), 0);
		send(clientSocket, reinterpret_cast<char*>(&rows), sizeof(rows), 0);
		send(clientSocket, reinterpret_cast<char*>(&cols), sizeof(cols), 0);

		std::vector<std::vector<char>> batch;
		for (int i = 0; i < BATCHSIZE && Iter != End; i++, Iter++)
		{
			std::vector<char> batchelement;
			batchelement = LoadBinaryData(Iter->path().string());
			std::cout << Iter->path().string() << std::endl;
			if (batchelement.size() > 0)
				batch.push_back(batchelement);
			else
			{
				std::cout << "Error in Loading Data. Exiting Program" << std::endl;
				return 0;
			}
		}
		if (batch.size() != 16)
		{
			while (batch.size() != 16)
				batch.push_back(batch[0]);
		}
		for (int i = 0; i < BATCHSIZE; i++)
		{
			if (send(clientSocket, (char*)batch[i].data(), batch[i].size(), 0) == -1)
				std::cout << "Error Sending" << std::endl;
			else
				std::cout << "Image Sent" << std::endl;
		}
	//}
	closesocket(clientSocket);
	closesocket(serverSocket);
	WSACleanup();
	return 0;
}