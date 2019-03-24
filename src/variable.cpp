#include "variable.h"
#include "operator.h"
//#include <iostream>
namespace dlframework{

Variable::Variable(bool does_require_grad)
{
	data=nullptr;grad=nullptr;op=-1;requires_grad=does_require_grad;
}

Variable::Variable(float x, bool does_require_grad)
{
	data=new Tensor(x); //when should we delete data ?
	grad=nullptr;
	op=-1;
	requires_grad=does_require_grad;
}

Variable::Variable(const std::initializer_list<unsigned> & init_shape, bool does_require_grad)
{
	data=new Tensor(init_shape);
	grad=nullptr;
	op=-1;
	requires_grad=does_require_grad;
}
Variable::Variable(const Variable & rhs) //Shadow Copy
{
	//std::cout<<"Variable copy constructor called"<<std::endl;
	data=rhs.data;
	grad=rhs.grad;
	op=rhs.op;
	requires_grad=rhs.requires_grad;
}
Variable::Variable(Variable && rhs)
{
	//std::cout<<"Variable move constructor called"<<std::endl;
	data=rhs.data;
	grad=rhs.grad;
	op=rhs.op;
	requires_grad=rhs.requires_grad;
	rhs.data=nullptr;
	rhs.grad=nullptr;
	rhs.op=-1;
	rhs.requires_grad=false;
}


Variable::Variable(Tensor & tensor, bool does_require_grad)
{
	//std::cout<<"Copy wrapper of a lvalue"<<std::endl;
	data=new Tensor(tensor);
	grad=nullptr;
	op=-1;
	requires_grad=does_require_grad;
}

Variable::Variable(Tensor && tensor, bool does_require_grad)
{
	//std::cout<<"Move wrapper of a rvalue"<<std::endl;
	data=new Tensor(std::move(tensor));
	grad=nullptr;
	op=-1;
	requires_grad=does_require_grad;
}

//void Variable::backward()
//{
//	if (data->length==1 && grad==nullptr)
//		grad=new Tensor(1);
	//else if grad==nullptr assert error
//	if (op!=nullptr)
//		op->bp(*this);
//}

Variable & Variable::operator=(const std::initializer_list<float> & array)
{
	if (data==nullptr) {// error
		
	}
	else
	{
		*data=array;
	}
	return *this;
}

void Variable::clear_data()
{
	if (data!=nullptr) delete data;
	data=nullptr;	
}

void Variable::zero_grad()
{
	if (grad!=nullptr)
		delete grad;
	grad=nullptr;
}

std::ostream& operator<<(std::ostream & o, const Variable & rhs)
{
	if (rhs.data!=nullptr)
		return o<<"Variable Containing "<<*rhs.data;
	else
		return o<<"Empty Data"<<std::endl;
}

Variable::~Variable()
{
	//std::cout<<"~Variable"<<std::endl;
	if (data!=nullptr) delete data;
	if (grad!=nullptr) delete grad;
}

} //end dlframework