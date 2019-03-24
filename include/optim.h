#ifndef OPTIM_H
#define OPTIM_H

#include "variable.h"
#include "functional.h"
#include <vector>

namespace dlframework{

class Optimizer{
public:
	static std::vector<Variable *> Params;
	int steps_count;
	float learning_rate;
	virtual void step()=0;
	Optimizer(float lr):steps_count{0},learning_rate{lr}{};
};

class SGDOptimizer:public Optimizer{
public:
	int batch_size;
	float weight_decay;
	std::vector<Tensor *> accumu_grads;
	SGDOptimizer(int bs, float lr=0.1, float wd=0);
	void step();
};




}//end namespace

#endif