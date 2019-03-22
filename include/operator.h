#ifndef OPERATOR_H
#define OPERATOR_H

#include "variable.h"


namespace dlframework{

class baseOp{
public:	
	int operand_num;
	Variable * operand[OPERATOR_MAX_NUM];
	baseOp(const std::initializer_list<Variable *> & operand_list);
	virtual Variable & cal()=0;
	virtual void bp(const Variable & res)=0;
	void _call_bp();
	virtual ~baseOp(){};
};

class op_Add:public baseOp{
public:
	op_Add(Variable & a, Variable & b);
	virtual Variable & cal();
	virtual void bp(const Variable & res);
	virtual ~op_Add(){};
};


} //end dlframework

#endif