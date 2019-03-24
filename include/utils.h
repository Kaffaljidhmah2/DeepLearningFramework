#ifndef UTILS_H
#define UTILS_H

#include "variable.h"
#include <string>
#include <fstream>
#include <ostream>


namespace dlframework{
namespace dataset{
	Tensor ** Read_MNIST_Train_Image(const char * url); //returns a pointer to a Tensor * array
	Tensor ** Read_MNIST_Train_Label(const char * url);
	Tensor ** Read_MNIST_Test_Iamge(const char * url);
	Tensor ** Read_MNIST_Test_Label(const char * url);

	Tensor ** _read_mnist_image(const char * url, int len);
	Tensor ** _read_mnist_label(const char * url, int len);

	void Visualize_Grayscale(const Tensor & , std::ostream &, float threshold=0.5);
}
}

#endif