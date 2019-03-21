#include "variable.h"

namespace dlframework{

template<class T> Tensor<T>::Tensor(std::initializer_list<unsigned> init_shape){
	dim = init_shape.end()-init_shape.begin();
	unsigned i=0;
	length=1;
	for (auto iter=init_shape.begin(); iter!= init_shape.end(); ++iter)
	{
		shape[i++]=*iter;
		length*=*iter;
	}
	shape[dim]=1;
	p=new T[length];
}

//overload all constructors !

template<class T> Tensor<T>::~Tensor(){
	delete[] p;	
}

template<class T> T& Tensor<T>::operator()(std::initializer_list<unsigned> indices){
	//assert dim == length of indices
	unsigned offset=0;
	unsigned i=0;
	for (auto iter=indices.begin(); iter!= indices.end(); ++iter)
	{
		offset+= *iter;
		offset*= shape[i+1];
		//assert *iter < shape[i]
		++i;
	}
	return p[offset];
}


}