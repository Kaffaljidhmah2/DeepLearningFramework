#ifndef OPERATOR_H
#define OPERATOR_H

#include "variable.h"


namespace dlframework{

class baseOp{
public:	
	int operand_num;
	Variable * operand[OPERATOR_MAX_NUM];
	Variable * result;
	baseOp(const std::initializer_list<Variable *> & operand_res_list);
	virtual void cal()=0;
	virtual void bp()=0;
	virtual ~baseOp(){};
};

// prefix with op_
class op_Add:public baseOp{
public:
	op_Add(Variable & a, Variable & b, Variable & res);
	virtual void cal();
	virtual void bp();
	virtual ~op_Add(){};
};


} //end dlframework

#endif