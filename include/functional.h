#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H

#include "variable.h"

namespace dlframework{
namespace functional{ //small letter

Tensor matmul(const Tensor & M, const Tensor & x);
Tensor add(const Tensor & a, const  Tensor & b);
Tensor sub(const Tensor & a, const Tensor & b);

Tensor max(const Tensor & a, const Tensor & b);
Tensor min(const Tensor & a, const Tensor & b);


}
}


#endif