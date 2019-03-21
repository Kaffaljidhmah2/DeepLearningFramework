#include "variable.h"
#include "operator.h"
#include <iostream>

using namespace std;
using namespace dlframework;

int main()
{
	Tensor<float> a({2,2}),b({2,2});
	a={1,2,3,5.7};
	b={0,1,2,3};
	Variable va(a),vb(b);
	op_Add node(va,vb);
	Variable vc=node.cal();
	cout<<(*(Tensor<float>*)(vc.data))({0,0})<<endl;
	cout<<(*(Tensor<float>*)(vc.data))({0,1})<<endl;
	cout<<(*(Tensor<float>*)(vc.data))({1,0})<<endl;
	cout<<(*(Tensor<float>*)(vc.data))({1,1})<<endl;

	return 0;
}