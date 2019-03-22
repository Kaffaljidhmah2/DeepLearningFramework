#include "variable.h"
#include "operator.h"

namespace dlframework{

baseOp::baseOp(const std::initializer_list<Variable *> & operand_list)
{
	operand_num=operand_list.end()-operand_list.begin();
	unsigned i=0;
	for (auto iter=operand_list.begin(); iter!=operand_list.end() ; ++iter,++i)
		operand[i]= *iter;
}

void baseOp::_call_bp()
{
	for (unsigned i=0;i<operand_num;++i)
		operand[i]->backward();
}

op_Add::op_Add(Variable & a,Variable & b):baseOp({&a,&b}){}

Variable & op_Add::cal()
{
	Variable *res=new Variable(*(operand[0]->data)+*(operand[1]->data));
	res->op=this;
	return *res;
}

void op_Add::bp(const Variable & res)
{
	if (operand[0]->grad==nullptr)
		operand[0]->grad = new Tensor(*res.grad);
	else
		*(operand[0]->grad) += *(res.grad);

	if (operand[1]->grad==nullptr)
		operand[1]->grad = new Tensor(*res.grad);
	else
		*(operand[1]->grad) += *(res.grad);
	_call_bp();
}


} //end dlframework