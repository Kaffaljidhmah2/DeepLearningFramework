#include "utils.h"

#include <iostream> //debug
namespace dlframework{
namespace dataset{

Tensor ** Read_MNIST_Train_Image(const char * url) //returns a pointer to a Tensor * array
{
	std::ifstream filereader(url,std::ios::in|std::ios::binary);
	if (!filereader)
	{
		std::cerr<<"Error "<<std::endl;
		//assert error.
	}
	Tensor ** res=new Tensor*[60000]; 
	unsigned char s;
	filereader.seekg(16,std::ios::beg);

	for (int image_id=0; image_id <60000 ; ++image_id)
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

} //end dataset
}
