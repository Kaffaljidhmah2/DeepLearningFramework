#include "variable.h"
#include "functional.h"
#include <algorithm>
#include <cmath>

namespace dlframework{
namespace functional{

Tensor matmul(const Tensor & M, const Tensor & x)
{
	//M=m*n;
	assert(M.dim==2);
	// x=n*k -> output m*k
	unsigned m=M.shape[0], n=M.shape[1];
	if (x.dim==2)
	{
		unsigned k=x.shape[1];
		assert(M.shape[1] == x.shape[0]);

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
		assert(M.shape[1] == x.shape[0]);

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
	assert(M.shape[1]==B.shape[1]);
	assert(M.dim==2);
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
	assert(M.shape[0]==A.shape[0]);
	assert(M.dim==2);
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
	assert(a.length==b.length);
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
	assert(a.length==b.length);
	Tensor res(a,true);
	for (int i=0;i<a.length;++i)
		res.p[i]=std::max(a.p[i],b.p[i]);
	return res;

}
Tensor min(const Tensor & a, const Tensor & b)
{
	//assert a.shape==b.shape
	assert(a.length==b.length);
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

Tensor drelu(const Tensor & input, const Tensor & grad)
{
	Tensor res(grad,true);
	for (int i=0;i<grad.length;++i)
		res.p[i]=(input.p[i]>0)?grad.p[i]:0;
	return res;
}

Tensor mean(const Tensor & x)
{
	Tensor res(0.0);
	for (int i=0;i<x.length;++i)
		res.p[0]+=x.p[i];
	res.p[0]/=x.length;
	return res;
}

Tensor sum(const Tensor & x)
{
	Tensor res(0.0);
	for (int i=0;i<x.length;++i)
		res.p[0]+=x.p[i];
	return res;
}

Tensor e_mul(const Tensor & x, const Tensor & y)
{
	//assert a.shape==b.shape
	assert(x.length==y.length);
	Tensor res(x,true);
	for (int i=0;i<x.length;++i)
	{
		res.p[i]=x.p[i]*y.p[i];
	}
	return res;
}

Tensor softmax(const Tensor & x)
{
	Tensor res(x,true);
	float & _to_del=*std::max_element(x.p, x.p+x.length);
	float Z=0.0;
	for (int i=0;i<x.length;++i)
	{
		res.p[i]=exp(x.p[i]-_to_del);
		Z += res.p[i];
	}
	for (int i=0;i<x.length;++i)
		res.p[i]/=Z;
	return res;
}

Tensor conv3d(const Tensor & im, const Tensor & kernel, int zero_pad, int stride)
{
	// im: in_channel * Height * Weight 
	// kernel: out_channel * in_channel * kernelsize1 * kernelsize2 
	// output: H*W -- zero_pad --> (H + 2zp) * (W + 2zp) -->  ceil_up (H + 2zp - k1 +1 )/stride * ceil_up(W + 2zp - k2 +1 )/stride 
	// output: out_channel * ( ) * ( )

	// assert dim and shape match
	assert(im.dim==3);
	assert(kernel.dim==4);
	assert(stride>0);
	assert(zero_pad>=0);

	const unsigned & in_channel = im.shape[0];
	const unsigned & H = im.shape[1];
	const unsigned & W = im.shape[2];
	const unsigned & out_channel = kernel.shape[0];
	const unsigned & k1= kernel.shape[2];
	const unsigned & k2= kernel.shape[3];

	assert(in_channel==kernel.shape[1]);

	Tensor res({out_channel, (H+2*zero_pad-k1+stride)/stride ,(W+2*zero_pad-k2+stride)/stride});

	for (int chan = 0; chan < out_channel ; ++chan)
	{
		for (int i=0;i<H+2*zero_pad-k1+1;i+=stride)
		{
			for (int j=0;j<W+2*zero_pad-k2+1;j+=stride)
			{
				float & current = res({chan, i/stride, j/stride});
				current=0.0;
				for (int inchan=0; inchan <in_channel ; ++inchan)
				{
					// current up_left corner of the kernel : (i-zero_pad, j-zero_pad)
					// current down_right corner : (i-zero_pad+k1, j-zero_pad+k2)
					for (int ii=0; ii<k1;++ii)
					{
						for (int jj=0;jj<k2;++jj)
						{
							float left=0.0;
							int right_i=i-zero_pad+ii;
							int right_j=j-zero_pad+jj;
							if (right_i >=0 && right_i<H && right_j>=0 && right_j<W)
								left = im({inchan, right_i, right_j});
							current += left * kernel({chan, inchan, ii, jj});
						}
					}
				}
			}
		}
	}
	return res;
}

}//end functional
}