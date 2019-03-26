#include "dmap.h"
namespace dlframework{

// Graph::Graph(){}

std::vector<baseOp*> Graph::op_stack; 
std::vector<Variable*> Graph::v_stack;

Variable & Graph::Add(Variable & a, Variable & b)
{
	v_stack.push_back(new Variable(true));
	op_stack.push_back(new op_Add(a,b,*v_stack.back()));
	v_stack.back()->op=op_stack.size()-1;
	return *v_stack.back();
}

Variable & Graph::Sub(Variable & a, Variable & b)
{
	v_stack.push_back(new Variable(true));
	op_stack.push_back(new op_Sub(a,b,*v_stack.back()));
	v_stack.back()->op=op_stack.size()-1;
	return *v_stack.back();
}

Variable & Graph::MatMul(Variable & a, Variable & b)
{
	v_stack.push_back(new Variable(true));
	op_stack.push_back(new op_MatMul(a,b,*v_stack.back()));
	v_stack.back()->op=op_stack.size()-1;
	return *v_stack.back();
}

Variable & Graph::InnerProduct(Variable & a, Variable & b)
{
	v_stack.push_back(new Variable(true));
	op_stack.push_back(new op_InnerProduct(a,b,*v_stack.back()));
	v_stack.back()->op=op_stack.size()-1;
	return *v_stack.back();
}

Variable & Graph::ReLU(Variable & a)
{
	v_stack.push_back(new Variable(true));
	op_stack.push_back(new op_ReLU(a,*v_stack.back()));
	v_stack.back()->op=op_stack.size()-1;
	return *v_stack.back();
}
Variable & Graph::SoftmaxCrossEntropy(Variable & x, Variable & label)
{
	v_stack.push_back(new Variable(true));
	op_stack.push_back(new op_SoftmaxCrossEntropy(x, label,*v_stack.back()));
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

std::vector<int> Graph::_get_subgraph(const Variable & z)
{
	std::queue<int> que_tmp;
	std::vector<int> cal_list;
	if (z.op!=-1)
	{
		que_tmp.push(z.op);
		cal_list.push_back(z.op);	
	}
	while(!que_tmp.empty())
	{
		int curr_op=que_tmp.front();que_tmp.pop();
		for (int i=0;i<op_stack[curr_op]->operand_num;++i)
		{
			int & next_op = op_stack[curr_op]->operand[i]->op;
			if (next_op!=-1)
			{
				cal_list.push_back(next_op);
				que_tmp.push(next_op);	
			}
		}
	}
	std::sort(cal_list.begin(), cal_list.end());
  	auto end=std::unique(cal_list.begin(), cal_list.end());
  	cal_list.resize(std::distance(cal_list.begin(),end)); 
  	return cal_list;
}

void Graph::eval(const Variable & z)	
{
	std::vector<int> cal_list=_get_subgraph(z);
	for (auto it=cal_list.begin();it!= cal_list.end();++it)
		op_stack[*it]->cal();
}

void Graph::backward(Variable & z)
{
	if (z.data->dim==1 && z.grad==nullptr)
	{
		z.grad=new Tensor(1);
	}
	//if (z.grad==nullptr) assert error!
	assert(z.grad!=nullptr);

	std::vector<int> cal_list=_get_subgraph(z);
	for (auto it=cal_list.rbegin();it!= cal_list.rend();++it)
		op_stack[*it]->bp();

}

void Graph::clear()
{
	op_stack.clear();
	v_stack.clear();
}

// Graph::~Graph()
// {
// 	op_stack.clear();
// 	v_stack.clear();
// }

}