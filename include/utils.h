#ifndef UTILS_H
#define UTILS_H

#include "variable.h"
#include <string>
#include <fstream>

namespace dlframework{
namespace dataset{
	Tensor ** Read_MNIST_Train_Image(const char * url); //returns a pointer to a Tensor * array
	Tensor ** Read_MNIST_Train_Label(const char * url);
	Tensor ** Read_MNIST_Test_Iamge(const char * url);
	Tensor ** Read_MNIST_Test_Label(const char * url);
}
}

#endif