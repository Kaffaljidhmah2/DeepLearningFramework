#ifndef VARIABLE_H
#define VARIABLE_H

#define TensorMaxDim 10

#include<initializer_list>

namespace dlframework{
	class BaseTensor{
	public:
		virtual ~BaseTensor(){};
	};

	template<class T>
	class Tensor:public BaseTensor{
	public:
		T * p;
		int dim;
		int length;
		int shape[TensorMaxDim];
		Tensor(std::initializer_list<unsigned>);
		T & operator()(std::initializer_list<unsigned>);
		virtual ~Tensor();
	};

//	class Variable{
//	public:
//		BaseTensor * data;
//		BaseTensor * grad;
//		void * op;
//		Variable();
//		void backward();
//		void zero_grad();
//
//	};

}// end dlframework

#endif