#include "variable.h"
#include <iostream>

using namespace std;
using namespace dlframework;

int main()
{
	Tensor<int> a({2,3,5});
	cout<<a.dim<<endl;
	cout<<a({1,2,3})<<endl;

}