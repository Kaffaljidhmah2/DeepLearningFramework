#ifndef VARIABLE_H
#define VARIABLE_H

#include<initializer_list>
#include<ostream>
#include<random>

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
		
		void reshape(const std::initializer_list<unsigned> & init_shape);
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

		Variable & operator=(const std::initializer_list<float> & array);

		void clear_data();
		void zero_grad();
		friend std::ostream& operator<<(std::ostream & o, const Variable & rhs);
		virtual ~Variable();
	};

	class Init{
	public:
		static std::default_random_engine dlframework_random_generator;
		static void set_seed(unsigned val);
		static void normal(Tensor & x, float mean=0.0, float std=1.0);
		static void uniform(Tensor & x, float a=0.0, float b=1.0);
	};


}// end dlframework

#endif