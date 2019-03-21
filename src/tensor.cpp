#include "variable.h"

namespace dlframework{

 Tensor::Tensor(const std::initializer_list<unsigned> & init_shape){
	dim = init_shape.end()-init_shape.begin();
	unsigned i=0;
	length=1;
	for (auto iter=init_shape.begin(); iter!= init_shape.end(); ++iter)
	{
		shape[i++]=*iter;
		length*=*iter;
	}
	shape[dim]=1;
	p=new float[length];
}

//overload all constructors !

 Tensor::Tensor(const Tensor & rhs)
{
	dim=rhs.dim;
	length=rhs.length;
	for (unsigned i=0;i<=dim;++i) shape[i]=rhs.shape[i];
	p=new float[length];
	for (unsigned i=0;i<length;++i) p[i]=rhs.p[i];
}

 Tensor & Tensor::operator=(const std::initializer_list<float> & array)
{
	float * ptr=p;
	for (auto it=array.begin(); it!=array.end() && ptr-p<length ; ++ptr,++it)
	{
		*ptr = *it;
	}
	return *this;
}

 Tensor::~Tensor(){
	delete[] p;	
}

float& Tensor::operator()(const std::initializer_list<unsigned> & indices){
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

 Tensor & Tensor::operator+(const Tensor& b)
{
	Tensor* x=new Tensor(*this);
	//assert shape match!
	for (unsigned i=0; i<length; ++i)
	{
		(x->p)[i]+=(b.p)[i];
	}
	return *x;
}
}