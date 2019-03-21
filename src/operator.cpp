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

op_Add::op_Add(Variable & a,Variable & b):baseOp({&a,&b}){}

Variable & op_Add::cal()
{
	Variable *res=new Variable(*(operand[0]->data)+*(operand[1]->data));
	return *res;
}


} //end dlframework