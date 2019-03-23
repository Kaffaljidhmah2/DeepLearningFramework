#include "dmap.h"
#include <iostream>
namespace dlframework{

Graph::Graph(){}


Variable & Graph::Add(Variable & a, Variable & b)
{
	v_stack.push_back(new Variable());
	op_stack.push_back(new op_Add(a,b,*v_stack.back()));
	v_stack.back()->op=op_stack.size()-1;
	return *v_stack.back();
}

Variable & Graph::Sub(Variable & a, Variable & b)
{
	v_stack.push_back(new Variable());
	op_stack.push_back(new op_Sub(a,b,*v_stack.back()));
	v_stack.back()->op=op_stack.size()-1;
	return *v_stack.back();
}

Variable & Graph::MatMul(Variable & a, Variable & b)
{
	v_stack.push_back(new Variable());
	op_stack.push_back(new op_MatMul(a,b,*v_stack.back()));
	v_stack.back()->op=op_stack.size()-1;
	return *v_stack.back();
}

Variable & Graph::InnerProduct(Variable & a, Variable & b)
{
	v_stack.push_back(new Variable());
	op_stack.push_back(new op_InnerProduct(a,b,*v_stack.back()));
	v_stack.back()->op=op_stack.size()-1;
	return *v_stack.back();
}

void Graph::zero_grad()
{
	for (auto it=op_stack.begin();it!=op_stack.end();++it)
	{
		for (int j=0;j<(*it)->operand_num;++j)
			(*it)->operand[j]->zero_grad();
	}
}

void Graph::eval(const Variable & z)	// To do: compute a subgraph that is necessary for z.
{
	for (int curr_op=0; curr_op<=z.op; ++curr_op)
	{
		op_stack[curr_op]->cal();
	}
}

void Graph::backward(Variable & z)
{
	if (z.data->dim==1 && z.grad==nullptr)
	{
		z.grad=new Tensor(1);
	}
	//if (z.grad==nullptr) assert error!
	for (int curr_op=z.op; curr_op>=0; --curr_op)
	{
		op_stack[curr_op]->bp();
	}

}

void Graph::clear()
{
	op_stack.clear();
	v_stack.clear();
}

Graph::~Graph()
{
	op_stack.clear();
	v_stack.clear();
}

}