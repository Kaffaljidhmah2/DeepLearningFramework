#include "variable.h"
#include "functional.h"
#include <algorithm>

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

Tensor matmul_T(const Tensor & M, const Tensor & B)
{
	// Calculate M * B^T
	// M=(m,n) B=(k,n) MB^T=(m,k)

	unsigned m=M.shape[0],n=M.shape[1],k=B.shape[0];
	//assert M.shape[1]=B.shape[1]
	//assert M.dim==2
	if (B.dim==2)
	{
		Tensor res({m,k});
		for (int i=0;i<m;++i)
		{
			for (int j=0;j<k;++j)
			{
				float & current=res.p[i*k+j];
				current=0.0;
				for (int l=0;l<n;++l)
					current+= M.p[i*n+l]*B.p[j*n+l];
			}
		}
		return res;
	}

}

Tensor T_matmul(const Tensor & A, const Tensor & M)
{
	// Calculate A^T * M
	// M=(m,n) A=(m,k) A^TM=(k,n)
	unsigned m=M.shape[0],n=M.shape[1],k=A.shape[1];
	//assert M.shape[0]=A.shape[0]
	//assert M.dim==2
	if (A.dim==2)
	{
		Tensor res({k,n});
		for (int i=0;i<k;++i)
		{
			for (int j=0;j<n;++j)
			{
				float & current=res.p[i*n+j];
				current=0.0;
				for (int l=0;l<m;++l)
					current+= M.p[l*n+j]*A.p[l*k+i];
			}
		}
		return res;
	}
}

Tensor inner_prod(const Tensor & a, const Tensor & b)
{
	//sum of Element-wize product

	//assert shape match 
	float r=0.0;
	for (int i=0;i<a.length;++i)
		r+=a.p[i]*b.p[i];
	return Tensor(r);
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
	Tensor res(a,true);
	for (int i=0;i<a.length;++i)
		res.p[i]=std::max(a.p[i],b.p[i]);
	return res;

}
Tensor min(const Tensor & a, const Tensor & b)
{
	//assert a.shape==b.shape
	Tensor res(a,true);
	for (int i=0;i<a.length;++i)
		res.p[i]=std::min(a.p[i],b.p[i]);
	return res;
}

Tensor neg(const Tensor & a)
{
	Tensor res(a,true);
	for (int i=0;i<a.length;++i)
		res.p[i]=-a.p[i];
	return res;
}

Tensor cmul(const float & c, const Tensor & rhs)
{
	Tensor res(rhs,true);
	for (int i=0;i<rhs.length;++i)
		res.p[i]=c*rhs.p[i];
	return res;	
}

Tensor relu(const Tensor & x)
{
	Tensor res(x,true);
	for (int i=0;i<x.length;++i)
		res.p[i]=std::max(x.p[i],0.f);
	return res;
}

}//end functional
}