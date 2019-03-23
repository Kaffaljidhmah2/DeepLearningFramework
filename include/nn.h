#ifndef NN_H
#define NN_H

#include "dmap.h"
#include "functional.h"

namespace dlframework{
namespace nn{
	class Linear	//Variable container, and a graph builder
	{
	public:
		Variable weight;
		Variable bias;
		bool is_bias;
		Linear(unsigned in_dim, unsigned out_dim, bool has_bias=true);
		Variable & operator()(Variable &);
		//~Linear(); ?
	};

}
}


#endif