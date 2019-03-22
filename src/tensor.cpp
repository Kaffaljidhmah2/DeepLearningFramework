#include "variable.h"
#include "operator.h"

namespace dlframework{

 Tensor::Tensor(float x){
 	dim=1;
 	length=1;
 	shape[0]=1;
 	shape[1]=1;
 	p=new float[1];
 	p[0]=x;
 }

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

std::ostream& operator<<(std::ostream & o, const Tensor & rhs)
{
	for (int i=0;i<rhs.length;++i)
		o<<rhs.p[i]<<' ';
	return o;
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

Tensor & Tensor::operator+=(const Tensor & b)
{
	//assert shape match!
	for (unsigned i=0; i<length; ++i)
	{
		p[i]+=b.p[i];
	}
	return *this;
}

Tensor & Tensor::operator+(const Tensor& b)
{
	Tensor* x=new Tensor(*this);
	(*x)+=b;
	return *x;
}
}