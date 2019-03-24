#ifndef VARIABLE_H
#define VARIABLE_H

#include<initializer_list>
#include<ostream>

namespace dlframework{	

	const unsigned TENSOR_MAX_DIM=10;
 	const unsigned OPERATOR_MAX_NUM=10;
	class Tensor{
	public:
		int dim;
		int length;
		int shape[TENSOR_MAX_DIM];
		float * p;

		Tensor();
		Tensor(float);
		Tensor(const std::initializer_list<unsigned> & init_shape);
		Tensor(const Tensor & rhs);
		Tensor(const Tensor & rhs, bool shape_only);
		Tensor(Tensor && rhs); 
		virtual ~Tensor();

		Tensor & operator=(const std::initializer_list<float> & array);
		Tensor & operator=(const Tensor & rhs); 
		Tensor & operator=(Tensor && rhs);
		float & operator()(const std::initializer_list<unsigned> & indices);
		Tensor & operator+=(const Tensor & b);
		Tensor operator+(const Tensor& b) const;
		Tensor & operator-=(const Tensor & b);
		Tensor operator-(const Tensor & b) const;
		Tensor & operator*=(const float & x);
		
		friend std::ostream& operator<<(std::ostream & o, const Tensor & rhs);
	};


	class Variable{
	public:
		Tensor * data;
		Tensor * grad;
		bool requires_grad; //existing data: false; trainable parameters & intermediate results: true
		int op;

		Variable(bool does_require_grad=false);
		Variable(float x, bool does_require_grad=false);
		Variable(const std::initializer_list<unsigned> & init_shape, bool does_require_grad=false);
		Variable(const Variable & rhs);	//shadow copy
		Variable(Variable && rhs);	
		Variable(Tensor & tensor, bool does_require_grad=false); //identity copy of a new tensor.
		Variable(Tensor && tensor, bool does_require_grad=false);

		void clear_data();
		void zero_grad();
		friend std::ostream& operator<<(std::ostream & o, const Variable & rhs);
		virtual ~Variable();
	};

	


}// end dlframework

#endif