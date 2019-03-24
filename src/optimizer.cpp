#include "optim.h"

namespace dlframework{
	std::vector<Variable *> Optimizer::Params;

SGDOptimizer::SGDOptimizer(int bs, float lr, float wd):Optimizer(lr),batch_size{bs},weight_decay{wd},accumu_grads(Optimizer::Params.size(), nullptr){}

void SGDOptimizer::step()
{
	// Accumulate gradients
	for (int i=0;i<Optimizer::Params.size();++i)
	{
		if (accumu_grads[i]==nullptr)
			accumu_grads[i]=new Tensor(*Optimizer::Params[i]->grad);
		else
			*accumu_grads[i] += *Optimizer::Params[i]->grad;
	}

	++steps_count;
	if (steps_count % batch_size ==0)
	{
		for (int i=0;i<Optimizer::Params.size();++i)
		{
			if (weight_decay !=0)
				*Optimizer::Params[i]->data *= (1 - learning_rate * weight_decay);
			*Optimizer::Params[i]->data -= functional::cmul(learning_rate / batch_size, *accumu_grads[i]);
			// zero accumu_grads	
			delete accumu_grads[i];
			accumu_grads[i]=nullptr;
		}
	}
}

}