#include "variable.h"

namespace dlframework{

Variable::Variable(BaseTensor & tensor)
{
	data=&tensor;
}

void Variable::backward()
{

}

void Variable::zero_grad()
{
	delete grad;
	grad=nullptr;
}


} //end dlframework