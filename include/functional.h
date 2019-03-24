#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H

#include "variable.h"

namespace dlframework{
namespace functional{ //small letter

Tensor matmul(const Tensor & M, const Tensor & x);
Tensor matmul_T(const Tensor & dY, const Tensor & B);
Tensor T_matmul(const Tensor & A, const Tensor & dY);
Tensor add(const Tensor & a, const  Tensor & b);
Tensor sub(const Tensor & a, const Tensor & b);
Tensor cmul(const float & c, const Tensor & rhs);

Tensor max(const Tensor & a, const Tensor & b);
Tensor min(const Tensor & a, const Tensor & b);
Tensor neg(const Tensor & a);

Tensor inner_prod(const Tensor & a, const Tensor & b);
Tensor relu(const Tensor & x);
Tensor drelu(const Tensor & input, const Tensor & grad);
Tensor mean(const Tensor & x);

}
}


#endif