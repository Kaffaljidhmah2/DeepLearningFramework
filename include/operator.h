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

class op_Sub:public baseOp{
public:
	op_Sub(Variable & a, Variable & b, Variable & res);
	virtual void cal();
	virtual void bp();
	virtual ~op_Sub(){};
};

class op_MatMul:public baseOp{
public:
	op_MatMul(Variable & a, Variable & b, Variable & res);
	virtual void cal();
	virtual void bp();
	virtual ~op_MatMul(){};
};

class op_InnerProduct:public baseOp{
public:
	op_InnerProduct(Variable & a, Variable & b, Variable & res);
	virtual void cal();
	virtual void bp();
	virtual ~op_InnerProduct(){};	
};


} //end dlframework

#endif