#include "variable.h"
#include "operator.h"

namespace dlframework{

Variable::Variable(float x)
{
	data=new Tensor(x);
	grad=nullptr;
	op=nullptr;
}
Variable::Variable(const std::initializer_list<unsigned> & init_shape)
{
	data=new Tensor(init_shape);
	grad=nullptr;
	op=nullptr;
}
Variable::Variable(const Variable & rhs) //Shadow Copy
{
	data=rhs.data;
	grad=rhs.grad;
	op=rhs.op;
}

Variable::Variable(Tensor & tensor)
{
	data=&tensor;
	grad=nullptr;
	op=nullptr;
}

void Variable::backward()
{
	if (data->length==1 && grad==nullptr)
		grad=new Tensor(1);
	//else if grad==nullptr assert error
	if (op!=nullptr)
		op->bp(*this);
}

void Variable::zero_grad()
{
	if (grad!=nullptr)
		delete grad;
	grad=nullptr;
}

std::ostream& operator<<(std::ostream & o, const Variable & rhs)
{
	return o<<*rhs.data;
}

} //end dlframework