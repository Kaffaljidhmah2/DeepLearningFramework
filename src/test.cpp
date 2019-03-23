#include "variable.h"
#include "operator.h"
#include "functional.h"
#include <iostream>

using namespace std;
using namespace dlframework;

int main()
{
	// Tensors 

	
	// Tensor x,y(1.4),z({2,3}),w({2,3,4});
	// cout<<x<<endl;
	// cout<<y<<endl;
	// cout<<z<<endl;
	// cout<<w<<endl;

	//Deep copy of Tensors.
	// Tensor y(3.2);
	// Tensor x(y),z;
	// x({0})=0.2;cout<<y<<endl;
	// z=x; z({0})=1000;cout<<x<<endl;

	


	// // Assign Values, indices.
	// Tensor x({2,3});
	// Tensor y(x);
	// x={0,1,2,3,4,5};
	// y={2,3,4,5,9,1};
	// cout<<x<<endl<<y<<endl;
	// cout<<x({0,1})<<' '<<y({1,2})<<endl;
	// //Define a tensor of the same shape
	// Tensor z(x,true);
	// cout<<z<<endl;

	// //Basic Operations.

	// Tensor x({2,2}),y({2,2});
	// x={1,2,3,4}; y={2,3,4,1};
	// cout<<x+y<<endl;
	// cout<<x<<endl;
	// cout<<y<<endl;
	// x+=y;
	// cout<<x<<endl;
	// cout<<y<<endl;
	// y-=y;
	// cout<<x<<endl;
	// cout<<y<<endl;

	// //Variables
	// Variable x;cout<<x<<endl;
	// Variable y(1.2);cout<<y<<endl;
	// Variable z({2,3,4});cout<<z<<endl;

	//Shadow Copy
	// Variable y(1.2);cout<<y<<endl;
	// Variable w(y); w.clear_data(); cout<<y<<endl; //Error !  > Use smart pointer!

	//Variable is a container of Tensor
	//Variable holds a deep copy of a tensor
	// Tensor tx({2,3});Variable vx(tx); cout<<vx<<endl;
	// tx={2,3,7};cout<<tx<<endl; cout<<vx<<endl;

	// //std::move
	// Variable v(Tensor({2,3}));

	// Functionals 

	// Tensor x({2,3}),y({2,3});x={1,2,2,1,2,2};y={2,-1,0,4,4,1};
	// cout<<x<<endl;cout<<y<<endl;
	// cout<<functional::add(x,y)<<endl;
	// cout<<functional::add(y,x)<<endl;
	// cout<<functional::sub(x,y)<<endl;
	// cout<<functional::sub(y,x)<<endl;
	// cout<<functional::add(functional::add(x,y),y)<<endl;

	// cout<<functional::max(x,y)<<endl;
	// cout<<functional::min(x,y)<<endl;

	// Tensor x({2,2}),y({2,2});x={1,2,2,1};y={1,0,0,1};
	// cout<<x<<endl;cout<<y<<endl;
	// cout<<functional::matmul(x,y)<<endl;
	// // 1 2  1 0   =   1 2 
	// // 2 1  0 1   =   2 1




	return 0;
}