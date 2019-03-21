#ifndef VARIABLE_H
#define VARIABLE_H

#define TENSOR_MAX_DIM 10
#define OPERATOR_MAX_NUM 10

#include<initializer_list>

namespace dlframework{	
	class Tensor{
	public:
		int dim;
		int length;
		int shape[TENSOR_MAX_DIM];
		float * p;
		Tensor(const std::initializer_list<unsigned> & init_shape);
		Tensor(const Tensor & rhs);
		Tensor & operator=(const std::initializer_list<float> & array);
		float & operator()(const std::initializer_list<unsigned> & indices);
		Tensor & operator+(const Tensor& b);
		virtual ~Tensor();
	};

	class baseOp;
	class Variable{
	public:
		Tensor * data;
		Tensor * grad;
		baseOp * op;
		Variable(Tensor & tensor);
		void backward();
		void zero_grad();
	};

	


}// end dlframework

#endif