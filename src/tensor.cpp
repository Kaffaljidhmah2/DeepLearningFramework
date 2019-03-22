#include "variable.h"
#include "operator.h"
#include <iostream> //debug
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

Tensor::Tensor()
{

	dim=0;length=0;p=nullptr;shape[0]=1;
}

 Tensor::Tensor(const Tensor & rhs)	//shadow copy
{
	dim=rhs.dim;
	length=rhs.length;
	for (unsigned i=0;i<=dim;++i) shape[i]=rhs.shape[i];
	p=rhs.p;
}

void Tensor::copy(const Tensor & rhs)
{
	dim=rhs.dim;
	length=rhs.length;
	for (unsigned i=0;i<=dim;++i) shape[i]=rhs.shape[i];
	if (p!=nullptr) delete[] p;
	p=new float[length];
	for (unsigned i=0;i<length;++i) p[i]=rhs.p[i];
}

std::ostream& operator<<(std::ostream & o, const Tensor & rhs)
{
	o<<"Tensor with dim="<<rhs.dim<<", shape=(";
	for (int i=0;i<rhs.dim;++i) o<<rhs.shape[i]<<',';
	o<<")"<<std::endl;
	for (int i=0;i<rhs.length;++i)
		{o<<rhs.p[i]<<' ';}
	return o;
}


 Tensor & Tensor::operator=(const std::initializer_list<float> & array) //list init
{
	//Assert size_mismatch
	float * ptr=p;
	for (auto it=array.begin(); it!=array.end() && ptr-p<length ; ++ptr,++it)
	{
		*ptr = *it;
	}
	return *this;
}

Tensor & Tensor::operator=(const Tensor & rhs)
{
	dim=rhs.dim;
	length=rhs.length;
	for (unsigned i=0;i<=dim;++i) shape[i]=rhs.shape[i];
	p=rhs.p;
	return *this;
}

Tensor::~Tensor(){
 	if (p!=nullptr) delete[] p;
 	std::cout<<"DEBUG INFO: DELETE Tensor with "<<length<<"DATA"<<std::endl;
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

Tensor Tensor::operator+(const Tensor& b) const
{
	Tensor x;
	x.copy(*this);
	x+=b;
	return x;
}

Tensor & Tensor::operator-=(const Tensor & b)
{
	//assert shape match!
	for (unsigned i=0; i<length; ++i)
	{
		p[i]-=b.p[i];
	}
	return *this;
}

Tensor Tensor::operator-(const Tensor& b) const
{
	Tensor x;
	x.copy(*this);
	x-=b;
	return x;
}
}