#ifndef VARIABLE_H
#define VARIABLE_H

#define TENSOR_MAX_DIM 10
#define OPERATOR_MAX_NUM 10

#include<initializer_list>

namespace dlframework{
	class BaseTensor{
	public:
		int dim;
		int length;
		int shape[TENSOR_MAX_DIM];
		void * p;
		virtual BaseTensor & operator+(const BaseTensor & b)=0;
		virtual ~BaseTensor(){};
	};

	template<class T>
	class Tensor:public BaseTensor{
	public:
		Tensor(const std::initializer_list<unsigned> & init_shape);
		Tensor(const Tensor<T> & rhs);
		Tensor<T> & operator=(const std::initializer_list<T> & array);
		T & operator()(const std::initializer_list<unsigned> & indices);
		Tensor<T> & operator+(const BaseTensor& b);
		virtual ~Tensor();
	};


	class baseOp;
	class Variable{
	public:
		BaseTensor * data;
		BaseTensor * grad;
		baseOp * op;
		Variable(BaseTensor & tensor);
		void backward();
		void zero_grad();
	};

	


}// end dlframework

#include <tensor.tpp>

#endif