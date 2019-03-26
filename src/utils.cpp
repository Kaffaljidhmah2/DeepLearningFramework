#include "utils.h"

#include <iostream> //debug
namespace dlframework{
namespace dataset{

Tensor ** _read_mnist_image(const char * url, int len)
{
	std::ifstream filereader(url,std::ios::in|std::ios::binary);
	// if (!filereader)
	// {
	// 	std::cerr<<"Error "<<std::endl;
	// 	//assert error.
	// }
	assert(filereader);
	Tensor ** res=new Tensor*[len]; 
	unsigned char s;
	filereader.seekg(16,std::ios::beg);

	for (int image_id=0; image_id <len ; ++image_id)
	{
		res[image_id]=new Tensor({28,28});
		for (int row=0;row<28;++row)
		{
			for (int col=0;col<28;++col)
			{
				filereader.read((char*) &s , 1);
				res[image_id]->p[row*28+col] = ((float)s)/256.0;
			}
		}
	}
	filereader.close();
	return res;	
}

Tensor ** _read_mnist_label(const char * url, int len)
{
	std::ifstream filereader(url,std::ios::in|std::ios::binary);
	if (!filereader)
	{
		std::cerr<<"Error "<<std::endl;
		//assert error.
	}
	Tensor ** res=new Tensor*[len]; 
	unsigned char s;
	filereader.seekg(8,std::ios::beg);

	for (int image_id=0; image_id <len ; ++image_id)
	{
		filereader.read((char*) &s , 1);
		res[image_id]=new Tensor((float)s);
	}
	filereader.close();
	return res;	
}

Tensor ** Read_MNIST_Train_Image(const char * url) //returns a pointer to a Tensor * array
{
	return _read_mnist_image(url, 60000);
}

Tensor ** Read_MNIST_Test_Image(const char * url)
{
	return _read_mnist_image(url, 10000);
}

Tensor ** Read_MNIST_Train_Label(const char * url)
{
	return _read_mnist_label(url,60000);
}

Tensor ** Read_MNIST_Test_Label(const char * url)
{
	return _read_mnist_label(url,10000);
}

void Visualize_Grayscale(const Tensor & x, std::ostream & o, float threshold)
{
	if (x.dim==2)
	{
		for (unsigned i=0;i<x.shape[0];++i)
		{
			for (unsigned j=0;j<x.shape[1];++j)
			{
				if (x.p[i*x.shape[1]+j] > threshold)
					o<<'*';
				else
					o<<' ';
			}
			o<<std::endl;
		}
	}
	else
	{
		//assert
		//error
	}
}

} //end dataset
}
