#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include<random>

#pragma comment(lib, "ws2_32.lib")

#define PREVPORT 13537
#define NEXTPORT 13538
#define KERNELS 16
#define KERNELSIZE 5
#define POOLSIZE 2
#define LR 0.0001
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


void UpdateBias(std::vector<float>& bias, const std::vector < std::vector<std::vector<float>>>& gradients)
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
		bias[i] -= LR * temp[i];
	}
}

std::vector<std::vector<std::vector<float>>> UnPool(const std::vector<std::vector<std::vector<float>>>& gradients,
	const std::vector<std::vector<std::vector<float>>>& pooled,const std::vector<std::vector<std::vector<float>>> &  convolved)
{
	std::vector<std::vector<std::vector<float>>> Unpooled;
	Unpooled.resize(convolved.size());
	for (int i = 0; i < convolved.size(); i++)
	{
		Unpooled[i].resize(convolved[i].size());
		for (int j = 0; j <= convolved[i].size() - POOLSIZE; j+=POOLSIZE)
		{
			for (int k = 0; k <= convolved[i][j].size() - POOLSIZE; k+=POOLSIZE)
			{
				for (int l = j; l < j + POOLSIZE; l++)
				{
					for (int m = k; m < k + POOLSIZE; m++)
					{
						if (pooled[i][j/2][k/2] != convolved[i][l][m] or pooled[i][j/2][k/2] <= 0)//Derivative of ReLU
							Unpooled[i][l].push_back(0);
						else
							Unpooled[i][l].push_back(gradients[i][j/2][k/2]);
					}
				}
				
			}
		}
	}
	return Unpooled;
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

std::vector<std::vector<float>> initKernels(int numofprevkernels)//He initialization
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> distributer(0.0, std::sqrt(2.0 / (KERNELSIZE * KERNELSIZE) * numofprevkernels));
	std::vector<std::vector<float>> weights;
	for (int i = 0; i < KERNELSIZE; i++)
	{
		weights.push_back(std::vector<float>());
		for (int j = 0; j < KERNELSIZE; j++)
			weights.at(i).push_back(distributer(gen)*0.1);
	}
	return weights;
}

float dotproduct(const std::vector<std::vector<float>>& image, const std::vector<std::vector<float>>& kernel, int row, int col)//Dot Product
{
	float sum = 0;
	for (int i = row; i < row + kernel.size(); i++)
	{
		for (int j = col; j < col + kernel.size(); j++)
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
float Maxwindow(const std::vector<std::vector<float>>& matrix, int row, int col)//Function to find max value in POOLSIZExPOOLSIZE window
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
		for (int j = 0; j <= ConvolvedImages[i].size() - POOLSIZE; j += POOLSIZE)
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
	const std::vector<std::vector<std::vector<float>>>& kernels, const std::vector<float>& biases)
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

std::vector<std::vector<float>> FlipKernel(const std::vector<std::vector<float>>& kernel)
{
	std::vector<std::vector<float>> flipped;
	for (int i = kernel.size() - 1; i >= 0; i--)
	{
		flipped.push_back(std::vector<float>());
		for (int j = kernel[i].size() - 1; j >= 0; j--)
		{
			flipped[flipped.size() - 1].push_back(kernel[i][j]);
		}
	}
	return flipped;
}

std::vector<std::vector<float>> TransposedConvolution(const std::vector<std::vector<float>>& matrix, const std::vector < std::vector<float>>& kernel)
{
	std::vector<std::vector<float>> result(matrix.size() + kernel.size() - 1,std::vector<float>(matrix.size() + kernel.size() - 1,0));
	for (int i = 0; i < matrix.size(); i++)
	{
		for (int j = 0; j < matrix[0].size(); j++)
		{
			for (int k = 0; k < kernel.size(); k++)
			{
				for (int l = 0; l < kernel[0].size(); l++)
				{
					result[i+k][j+l] += kernel[k][l] * matrix[i][j];
				}
			}
		}
	}
	return result;
}

void UpdateWeights(std::vector<std::vector<std::vector<float>>>& kernels, const std::vector<std::vector<std::vector<float>>>& gradients,
					const std::vector<std::vector<std::vector<float>>>& inputs)
{
	std::vector<std::vector<std::vector<float>>> deltaw(KERNELS,std::vector<std::vector<float>>(KERNELSIZE,std::vector<float>(KERNELSIZE,0)));
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
				deltaw[i][j][k] *= LR;
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
	int batchsize, numofimages, rows, cols, prevlayerkernels;
	recv(prevsock, reinterpret_cast<char*>(&batchsize), sizeof(batchsize), 0);
	recv(prevsock, reinterpret_cast<char*>(&numofimages), sizeof(numofimages), 0);
	recv(prevsock, reinterpret_cast<char*>(&rows), sizeof(rows), 0);
	recv(prevsock, reinterpret_cast<char*>(&cols), sizeof(cols), 0);
	recv(prevsock, reinterpret_cast<char*>(&prevlayerkernels), sizeof(prevlayerkernels), 0);
	std::vector<std::vector<float>> featuremaps;
	std::vector<std::vector<std::vector<float>>> kernels, input,convolved,pooled;//Vector to store all the kernels,input,convolvedimages and pooled images
	std::vector<float> biases;//Vector to store all the biases
	for (int i = 0; i < KERNELS; i++)
	{
		kernels.push_back(initKernels(prevlayerkernels));
		biases.push_back(0);
	}
	int newrows = (rows - KERNELSIZE + 1) / POOLSIZE;
	int newcols = newrows;
	int numofkernels = KERNELS;
	int newnumofimages = numofimages * numofkernels;
	send(nextsock, reinterpret_cast<char*>(&batchsize), sizeof(batchsize), 0);
	send(nextsock, reinterpret_cast<char*>(&newnumofimages), sizeof(newnumofimages), 0);
	send(nextsock, reinterpret_cast<char*>(&newrows), sizeof(newrows), 0);
	send(nextsock, reinterpret_cast<char*>(&newcols), sizeof(newcols), 0);
	std::vector<std::vector<float>> BBoxCoords;
	std::vector<float>data(rows * cols);
	char temp;
	int numofBBox;
	while (recv(prevsock, &temp, 1, MSG_PEEK) != 0)
	{
		for (int i = 0; i < batchsize; i++)
		{
			recv(prevsock, reinterpret_cast<char*>(&numofBBox), sizeof(numofBBox), 0);//Recieving number of BBox in this image
			std::cout << numofBBox << " BBoxes Recieved" << std::endl;

			for (int j = 0; j < numofBBox; j++)
			{
				BBoxCoords.push_back(std::vector<float>(5));
				recv(prevsock, reinterpret_cast<char*>(BBoxCoords[j].data()), BBoxCoords[j].size() * sizeof(float), 0);//Recieving BBox Coords
			}

			for (int j = 0; j < numofimages; j++)
			{
				recv(prevsock, reinterpret_cast<char*>(data.data()), rows * cols * sizeof(float), 0);//Recieving Image Data
				input.push_back(ReconstructMatrix(data, rows, cols));
			}
			convolved = Convolution(input, kernels, biases);
			pooled = MaxPooling(convolved);
			numofBBox = BBoxCoords.size();
			send(nextsock, reinterpret_cast<char*>(&numofBBox), sizeof(numofBBox), 0);//Sending number of BBoxes in the image
			for (int j = 0; j < numofBBox; j++)
				send(nextsock, reinterpret_cast<char*>(BBoxCoords[j].data()), BBoxCoords[j].size() * sizeof(float), 0);//Sending BBox Coordinates

			for (int j = 0; j < newnumofimages; j++)
			{
				std::vector<float>image = FlattenMatrix(pooled[j]);
				if (send(nextsock, (char*)image.data(), image.size() * sizeof(float), 0) == -1)//Sending Image Data
					std::cout << "Error Sending Image" << std::endl;
				/*else
					std::cout << "Image" << j << " Sent" << std::endl;*/

			}

		}
		int outrows, outcols;
		outrows = pooled[0].size();
		outcols = pooled[0][0].size();
		std::vector<float> nextlayergradient(outrows*outcols), gradient;
		std::vector<std::vector<std::vector<float>>> unpooledgradients;
		for (int j = 0; j < newnumofimages; j++)
		{
			recv(nextsock, reinterpret_cast<char*>(nextlayergradient.data()), nextlayergradient.size()* sizeof(float), 0);//Recieving Gradients
			unpooledgradients.push_back(ReconstructMatrix(nextlayergradient, outrows, outcols));
		}
		unpooledgradients = UnPool(unpooledgradients, pooled,convolved);//Unpooling + ReLU derivative
		UpdateBias(biases,unpooledgradients);
		UpdateWeights(kernels, unpooledgradients, input);
		std::vector<std::vector<std::vector<float>>> flippedkernels,prevlayergradients;
		for (int i = 0; i < KERNELS; i++)
			flippedkernels.push_back(FlipKernel(kernels[i]));
		for (int i = 0; i < unpooledgradients.size(); i+=KERNELS)
		{
			prevlayergradients.push_back(TransposedConvolution(unpooledgradients[i], flippedkernels[0]));
			for (int j = 1; j < KERNELS; j++)
				prevlayergradients[i/KERNELS] = AddMatrices(prevlayergradients[i/KERNELS], TransposedConvolution(unpooledgradients[i+j],flippedkernels[j]));
		}

		for (int j = 0; j < prevlayergradients.size(); j++)
		{
			std::vector<float>image = FlattenMatrix(prevlayergradients[j]);
			if (send(prevsock, (char*)image.data(), image.size() * sizeof(float), 0) == -1)//Sending Image Data
				std::cout << "Error Sending Image" << std::endl;

		}
		std::cout << "Back Propagating..." << std::endl;
		BBoxCoords.clear();
		input.clear();
		convolved.clear();
		pooled.clear();
	}

	closesocket(nextsock);
	closesocket(prevsock);
	WSACleanup();
	return 0;
}