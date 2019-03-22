#include "variable.h"
#include "operator.h"

namespace dlframework{

baseOp::baseOp(const std::initializer_list<Variable *> & operand_res_list)
{
	operand_num=operand_res_list.end()-operand_res_list.begin()-1;
	unsigned i=0;
	for (auto iter=operand_res_list.begin(); (iter+1)!=operand_res_list.end() ; ++iter,++i)
		operand[i]= *iter;
	result=*(operand_res_list.end()-1);
}

op_Add::op_Add(Variable & a,Variable & b, Variable & res):baseOp({&a,&b, &res}){}

void op_Add::cal()
{
	result->clear_data();
	result->data=new Tensor(*(operand[0]->data) + *(operand[1]->data));
}

void op_Add::bp()
{
	if (operand[0]->grad==nullptr)
		operand[0]->grad = new Tensor(*result->grad);
	else
		*(operand[0]->grad) += *(result->grad);

	if (operand[1]->grad==nullptr)
		operand[1]->grad = new Tensor(*result->grad);
	else
		*(operand[1]->grad) += *(result->grad);
}


} //end dlframework