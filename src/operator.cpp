#include "variable.h"
#include "operator.h"
#include "functional.h"
#include <iostream>

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
	if (operand[0]->requires_grad)
	{
		if (operand[0]->grad==nullptr)
			operand[0]->grad = new Tensor(*result->grad);
		else
			*(operand[0]->grad) += *(result->grad);		
	}
	
	if (operand[1]->requires_grad)
	{
		if (operand[1]->grad==nullptr)
			operand[1]->grad = new Tensor(*result->grad);
		else
			*(operand[1]->grad) += *(result->grad);
	}
}

op_Sub::op_Sub(Variable & a,Variable & b, Variable & res):baseOp({&a,&b, &res}){}

void op_Sub::cal()
{
	result->clear_data();
	result->data=new Tensor(*(operand[0]->data) - *(operand[1]->data));
}

void op_Sub::bp()
{
	if (operand[0]->requires_grad)
	{	
		if (operand[0]->grad==nullptr)
			operand[0]->grad = new Tensor(*result->grad);
		else
			*(operand[0]->grad) += *(result->grad);
	}

	if (operand[1]->requires_grad)
	{
		if (operand[1]->grad==nullptr)
			operand[1]->grad = new Tensor(functional::neg(*result->grad));
		else
			*(operand[1]->grad) -= *(result->grad);	
	}
	
}

op_MatMul::op_MatMul(Variable & a,Variable & b, Variable & res):baseOp({&a,&b, &res}){}

void op_MatMul::cal()
{
	result->clear_data();
	result->data=new Tensor(functional::matmul(*(operand[0]->data) , *(operand[1]->data)));
}

void op_MatMul::bp()
{
	if (operand[0]->requires_grad)
	{
		if (operand[0]->grad==nullptr) //Y=AB; dA=dYB^T; dB=A^TdY;
			operand[0]->grad = new Tensor(functional::matmul_T(*result->grad, *operand[1]->data));
		else
			*(operand[0]->grad) += functional::matmul_T(*result->grad, *operand[1]->data);	
	}
	if (operand[1]->requires_grad)
	{
		if (operand[1]->grad==nullptr)
			operand[1]->grad = new Tensor(functional::T_matmul( *operand[0]->data , *result->grad));
		else
			*(operand[1]->grad) += functional::T_matmul( *operand[0]->data , *result->grad);	
	}
	
}

op_InnerProduct::op_InnerProduct(Variable & a,Variable & b, Variable & res):baseOp({&a,&b, &res}){}

void op_InnerProduct::cal()
{
	result->clear_data();
	result->data=new Tensor(functional::inner_prod(*(operand[0]->data) , *(operand[1]->data)));
}

void op_InnerProduct::bp()
{
	if (operand[0]->requires_grad)
	{
		if (operand[0]->grad==nullptr) 
			operand[0]->grad = new Tensor(functional::cmul(result->grad->p[0],*operand[1]->data));
		else
			*(operand[0]->grad) += functional::cmul(result->grad->p[0],*operand[1]->data);
	}
	if (operand[1]->requires_grad)
	{
		if (operand[1]->grad==nullptr)
			operand[1]->grad = new Tensor(functional::cmul(result->grad->p[0], *operand[0]->data ));
		else
			*(operand[1]->grad) += functional::cmul(result->grad->p[0], *operand[0]->data );
	}
}


op_ReLU::op_ReLU(Variable & x, Variable & res):baseOp({& x, & res}){}

void op_ReLU::cal()
{
	result->clear_data();
	result->data=new Tensor(functional::relu(*operand[0]->data));
}
void op_ReLU::bp()
{
	if (operand[0]->requires_grad)
	{
		if (operand[0]->grad==nullptr)
			operand[0]->grad = new Tensor(functional::drelu(*operand[0]->data, *result->grad));
		else
			*(operand[0]->grad) += functional::drelu(*operand[0]->data, *result->grad);
	}
}

op_SoftmaxCrossEntropy::op_SoftmaxCrossEntropy(Variable & x, Variable & label, Variable & res):baseOp({&x, &label, &res}),_softmax_res{nullptr}{};
void op_SoftmaxCrossEntropy::cal()
{
	result->clear_data();
	_softmax_res=new Tensor(functional::softmax(*operand[0]->data));
	result->data=new Tensor(-log(_softmax_res->p[unsigned(operand[1]->data->p[0])]));	//numerial stablization?
}
void op_SoftmaxCrossEntropy::bp()
{
	if (operand[0]->requires_grad)
	{
		_softmax_res->p[unsigned(operand[1]->data->p[0])]-=1.0;
		if (operand[0]->grad==nullptr)
			operand[0]->grad = new Tensor(functional::cmul(result->grad->p[0] , *_softmax_res));
		else
			*(operand[0]->grad) += functional::cmul(result->grad->p[0] , *_softmax_res);
	}
	delete _softmax_res;
	_softmax_res=nullptr;
}

} //end dlframework