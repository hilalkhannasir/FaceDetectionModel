#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include<random>
#include<fstream>

#pragma comment(lib, "ws2_32.lib")

#define PREVPORT 13537
#define NEXTPORT 13538
#define KERNELS 16
#define KERNELSIZE 5
#define POOLSIZE 2
#define LR 0.001
#define ALPHA 0
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

std::vector<std::vector<std::vector<float>>> LoadWeights()
{
	std::ifstream reader("E:\\weightsandbiases\\Weights3.txt");
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
	std::ifstream reader("E:\\weightsandbiases\\Bias3.txt");
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
	std::ofstream writer("Weights3.txt");
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
	std::ofstream writer("Bias3.txt");
	for (int i = 0; i < bias.size(); i++)
		writer << bias[i] << ' ';
	writer.close();
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
							Unpooled[i][l].push_back(ALPHA);
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
	std::normal_distribution<float> distributer(0.0, std::sqrt(2.0 / (KERNELSIZE * KERNELSIZE * numofprevkernels)));
	std::vector<std::vector<float>> weights;
	for (int i = 0; i < KERNELSIZE; i++)
	{
		weights.push_back(std::vector<float>());
		for (int j = 0; j < KERNELSIZE; j++)
			weights.at(i).push_back(distributer(gen));
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
		return value*ALPHA;
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

//std::vector<std::vector<std::vector<float>>> Convolution(const std::vector<std::vector<std::vector<float>>>& matrices,
//	const std::vector<std::vector<std::vector<float>>>& kernels, const std::vector<float>& biases)
//{
//	std::vector<std::vector<std::vector<float>>> Convolvedimages;
//	for (int i = 0; i < matrices.size(); i++)//At each image
//	{
//		for (int l = 0; l < kernels.size(); l++)//Apply all kernels one by one
//		{
//			Convolvedimages.push_back(std::vector<std::vector<float>>());
//			for (int j = 0; j <= matrices.at(i).size() - KERNELSIZE; j++)//At each row
//			{
//				Convolvedimages[Convolvedimages.size() - 1].push_back(std::vector<float>());
//				for (int k = 0; k <= matrices.at(i).at(j).size() - KERNELSIZE; k++)//and each column
//				{
//					float value = dotproduct(matrices[i], kernels[l], j, k);
//					value += biases[l];//Add Bias value
//					Convolvedimages[Convolvedimages.size() - 1][j].push_back(ActivationFunction(value));//Apply Activation Function
//				}
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
					const std::vector<std::vector<std::vector<float>>>& inputs,int batchsize)
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
	std::cout << "Receiving Batchsize: " << batchsize << std::endl;
	recv(prevsock, reinterpret_cast<char*>(&numofimages), sizeof(numofimages), 0);
	std::cout << "Receiving number of input images: " << numofimages << std::endl;
	recv(prevsock, reinterpret_cast<char*>(&rows), sizeof(rows), 0);
	std::cout << "Receiving rows: " << rows << std::endl;
	recv(prevsock, reinterpret_cast<char*>(&cols), sizeof(cols), 0);
	std::cout << "Receiving cols: " << cols << std::endl;
	recv(prevsock, reinterpret_cast<char*>(&prevlayerkernels), sizeof(prevlayerkernels), 0);
	std::cout << "Receiving Previous Layer kernels: " << prevlayerkernels << std::endl;
	std::vector<std::vector<float>> featuremaps;
	std::vector<std::vector<std::vector<float>>> kernels, input,convolved,pooled;//Vector to store all the kernels,input,convolvedimages and pooled images
	std::vector<float> biases;//Vector to store all the biases
	for (int i = 0; i < KERNELS; i++)
	{
		kernels.push_back(initKernels(prevlayerkernels));
		biases.push_back(0);
	}
	//kernels = LoadWeights();
	//biases = LoadBias();
	int newrows = (rows - KERNELSIZE + 1) / POOLSIZE;
	int newcols = newrows;
	int numofkernels = KERNELS;
	int newnumofimages = numofimages * numofkernels;
	//int outputsize = newnumofimages * newrows * newcols;
	send(nextsock, reinterpret_cast<char*>(&batchsize), sizeof(batchsize), 0);
	//send(nextsock, reinterpret_cast<char*>(&outputsize), sizeof(outputsize), 0);
	send(nextsock, reinterpret_cast<char*>(&newnumofimages), sizeof(newnumofimages), 0);
	//send(nextsock, reinterpret_cast<char*>(&newrows), sizeof(newrows), 0);
	//send(nextsock, reinterpret_cast<char*>(&newcols), sizeof(newcols), 0);
	std::vector<std::vector<float>> BBoxCoords;
	std::vector<float>data(rows * cols);
	char temp;
	int numofBBox;
	while (recv(prevsock, &temp, 1, MSG_PEEK) != 0)
	{
		for (int i = 0; i < batchsize; i++)
		{
			recv(prevsock, reinterpret_cast<char*>(&numofBBox), sizeof(numofBBox), 0);//Recieving number of BBox in this image

			for (int j = 0; j < numofBBox; j++)
			{
				BBoxCoords.push_back(std::vector<float>(5));
				recv(prevsock, reinterpret_cast<char*>(BBoxCoords[j].data()), BBoxCoords[j].size() * sizeof(float), 0);//Recieving BBox Coords
			}

			std::vector<std::vector<std::vector<float>>> tempinput;
			for (int j = 0; j < numofimages; j++)
			{
				recv(prevsock, reinterpret_cast<char*>(data.data()), rows * cols * sizeof(float), 0);//Recieving Image Data
				tempinput.push_back(ReconstructMatrix(data, rows, cols));
			}
			std::vector<std::vector<std::vector<float>>> temp;
			float count = 0.0;
			for (int h = 0; h < tempinput.size(); h++)
			{
				for (int j = 0; j < KERNELS; j++)
				{
					temp.push_back(Convolve(tempinput[h], kernels[j]));//applying convolution
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
			}
			std::cout << "Percentage 0 Neurons: " << 1.0 * count / (temp.size() * temp[0].size() * temp[0][0].size()) * 100 << std::endl;
			convolved.insert(convolved.end(), temp.begin(), temp.end());
			temp = MaxPooling(temp);
			pooled.insert(pooled.end(), temp.begin(), temp.end());
			send(nextsock, reinterpret_cast<char*>(&numofBBox), sizeof(numofBBox), 0);//Sending number of BBoxes in the image
			std::cout << numofBBox << " BBoxes sent" << std::endl;

			for (int j = 0; j < numofBBox; j++)
				send(nextsock, reinterpret_cast<char*>(BBoxCoords[j].data()), BBoxCoords[j].size() * sizeof(float), 0);//Sending BBox Coordinates
			std::cout << BBoxCoords[0].size() << " coordinates sent" << std::endl;
			std::vector<float> GAP;
			for (int j = 0; j < newnumofimages; j++)
			{
				GAP.push_back(Sum(pooled[j + (i * newnumofimages)]));
				GAP[j] /= (pooled[0].size() * pooled[0].size());
			}
			send(nextsock, (char*)GAP.data(), GAP.size() * sizeof(float), 0);
			std::cout << "Sending data of size " <<newnumofimages << std::endl;
			input.insert(input.end(), tempinput.begin(), tempinput.end());
			BBoxCoords.clear();
		}
		int outrows = pooled[0].size();
		int outcols = outrows;
		std::vector<float> nextlayergradient(newnumofimages), gradient;
		std::vector<std::vector<std::vector<float>>> unpooledgradients;
		for (int j = 0; j < batchsize; j++)
		{
			recv(nextsock, reinterpret_cast<char*>(nextlayergradient.data()), nextlayergradient.size()* sizeof(float), 0);//Recieving Gradients
			for (int l = 0; l < nextlayergradient.size(); l++)
			{
				std::vector<std::vector<float>> tempkernel;
				for (int i = 0; i < outrows; i++)//Reversing GAP
				{
					tempkernel.push_back(std::vector<float>());
					for (int k = 0; k < outcols; k++)
						tempkernel[i].push_back(nextlayergradient[l]);
				}
				unpooledgradients.push_back(tempkernel);
			}
		}
		unpooledgradients = UnPool(unpooledgradients, pooled,convolved);//Unpooling + ReLU derivative
		UpdateBias(biases,unpooledgradients,batchsize);
		UpdateWeights(kernels, unpooledgradients, input,batchsize);
		StoreWeights(kernels);
		StoreBias(biases);
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
		input.clear();
		convolved.clear();
		pooled.clear();
	}

	closesocket(nextsock);
	closesocket(prevsock);
	WSACleanup();
	return 0;
}