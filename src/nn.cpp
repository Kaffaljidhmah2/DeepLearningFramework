#include "nn.h"

namespace dlframework{
namespace nn{

Linear::Linear(unsigned in_dim, unsigned out_dim, bool has_bias):weight({out_dim,in_dim},true),is_bias(has_bias)
{
	if (is_bias)
	{
		bias.data=new Tensor({out_dim,1}); // ?
		bias.grad=nullptr;
		bias.op=-1;
		bias.requires_grad=true;
	}
	// initialize weight matrix!
	Optimizer::Params.push_back(&weight);
	Optimizer::Params.push_back(&bias);
}

Variable & Linear::operator()(Variable & input)
{
	Variable & mx = Graph::MatMul(weight,input);
	if (is_bias) 
	{
		Variable & mx_b = Graph::Add(mx,bias);
		return mx_b;
	}
	else return mx;
}

}//end namespace nn
}