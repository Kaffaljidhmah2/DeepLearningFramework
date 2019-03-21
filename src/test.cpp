#include "variable.h"
#include <iostream>

using namespace std;
using namespace dlframework;

int main()
{
	Tensor<float> a({1,2,2});
	a={1,2,3,5.7};
	cout<<a.dim<<endl;
	cout<<a({0,0,0})<<endl;
	cout<<a({0,0,1})<<endl;
	cout<<a({0,1,0})<<endl;
	cout<<a({0,1,1})<<endl;

	return 0;
}