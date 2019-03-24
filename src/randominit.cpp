#include "variable.h"

namespace dlframework{

std::default_random_engine Init::dlframework_random_generator(1);
void Init::set_seed(unsigned val)
{
	dlframework_random_generator.seed(val);
}
void Init::normal(Tensor & x, float mean, float std)
{
	std::normal_distribution<float> dist(mean,std);
	for (int i=0;i<x.length;++i)
	{
		x.p[i]=dist(dlframework_random_generator);
	}
}
}