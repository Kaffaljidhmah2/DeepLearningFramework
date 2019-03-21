#include "variable.h"

namespace dlframework{

template<class T> Tensor<T>::Tensor(const std::initializer_list<unsigned> & init_shape){
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

template<class T> Tensor<T>::Tensor(const Tensor<T> & rhs)
{
	dim=rhs.dim;
	length=rhs.length;
	for (unsigned i=0;i<=dim;++i) shape[i]=rhs.shape[i];
	p=new T[length];
	for (unsigned i=0;i<length;++i) ((T*)p)[i]=((T*)rhs.p)[i];
}

template<class T> Tensor<T> & Tensor<T>::operator=(const std::initializer_list<T> & array)
{
	T * ptr=(T*)p;
	for (auto it=array.begin(); it!=array.end() && ptr-(T*)p<length ; ++ptr,++it)
	{
		*ptr = *it;
	}
	return *this;
}

template<class T> Tensor<T>::~Tensor(){
	delete[] (T*)p;	
}

template<class T> T& Tensor<T>::operator()(const std::initializer_list<unsigned> & indices){
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
	return ((T*)p)[offset];
}

template<class T> Tensor<T> & Tensor<T>::operator+(const BaseTensor& b)
{
	Tensor<T>* x=new Tensor<T>(*this);
	//assert shape match!
	for (unsigned i=0; i<length; ++i)
	{
		((T*)x->p)[i]+=((T*)b.p)[i];
	}
	return *x;
}
}