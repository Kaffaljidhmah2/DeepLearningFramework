#include "variable.h"
#include "functional.h"

namespace dlframework{
namespace functional{

Tensor matmul(const Tensor & M, const Tensor & x)
{
	//assert M.dim==2; M=m*n;
	// x=n*k -> output m*k
	unsigned m=M.shape[0], n=M.shape[1];
	if (x.dim==2)
	{
		unsigned k=x.shape[1];
		//assert M.shape[1] == x.shape[0]

		Tensor res({m,k});
		for (unsigned i=0;i<m;++i)
		{
			for (unsigned j=0;j<k;++j)
			{
				float & current = res.p[i*k+j];
				current=0.0;
				for (unsigned l=0;l<n;++l)
				{
					current += M.p[i*n+l] * x.p[l*k+j];
				}
			}
		}
		return res;
	}
	else if (x.dim==1) //M: m*n x: n  -> output m
	{
		//assert M.shape[1] == x.shape[0]

		Tensor res({m});
		for (unsigned i=0;i<m;++i)
		{
			float & current = res.p[i];
			current=0;
			for (unsigned l=0;l<n;++l)
				current += M.p[i*n+l] * x.p[l];
		}
		return res;
	}

}

Tensor add(const Tensor & a, const  Tensor & b)
{
	return a+b;
}

Tensor sub(const Tensor & a, const Tensor & b)
{
	return a-b;
}

Tensor max(const Tensor & a, const Tensor & b)
{
	//assert a.shape==b.shape
	Tensor res(a);
	for (int i=0;i<a.length;++i)
		if (res.p[i]<b.p[i]) res.p[i]=b.p[i];
	return res;

}
Tensor min(const Tensor & a, const Tensor & b)
{
	//assert a.shape==b.shape
	Tensor res(a);
	for (int i=0;i<a.length;++i)
		if (res.p[i]>b.p[i]) res.p[i]=b.p[i];
	return res;
}

}//end functional
}