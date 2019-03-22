#include "variable.h"
#include "operator.h"
#include <iostream>

using namespace std;
using namespace dlframework;

int main()
{
//	Tensor a({2,2}),b({2,2});
//	a={1,2,3,5.7};
//	b={0,1,2,3};
//	Variable va(a),vb(b);
//	op_Add node(va,vb);
//	Variable vc=node.cal();

	Variable va(10),vb(10);
	op_Add gen_vc(va,va);
	Variable vc=gen_vc.cal();
	op_Add gen_vd(va,vc);
	Variable vd=gen_vd.cal();
	va.zero_grad();
	vb.zero_grad();
	vc.zero_grad();
	vd.zero_grad();
	vd.backward();

	cout<<(*(va.grad))<<endl;
	//cout<<(*(vb.grad))<<endl;
	cout<<(*(vc.grad))<<endl;
	cout<<(*(vd.grad))<<endl;
	cout<<va<<vb<<vc<<vd<<endl;


//	cout<<(*(vc.data))({0,0})<<endl;
//	cout<<(*(vc.data))({0,1})<<endl;
//	cout<<(*(vc.data))({1,0})<<endl;
//	cout<<(*(vc.data))({1,1})<<endl;

	return 0;
}