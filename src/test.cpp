#include "variable.h"
#include "operator.h"
#include "functional.h"
#include "dmap.h"
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

	// Variable is a container of Tensor
	// Variable holds a deep copy of a tensor
	// Tensor tx({2,3});Variable vx(tx); cout<<vx<<endl;
	// tx={2,3,7};cout<<tx<<endl; cout<<vx<<endl;

	// // To avoid deep copy, use std::move instead !

	// Variable vx_(std::move(tx));
	// cout<<tx<<endl;

	// //std::move
	// Variable v(Tensor({2,3}));

	// Functionals 

	// Tensor x({2,3}),y({2,3});x={1,2,2,1,2,2};y={2,-1,0,4,4,1};
	// cout<<x<<endl;cout<<y<<endl;
	// cout<<functional::relu(y)<<endl;
	// //cout<<functional::add(x,y)<<endl;
	// Tensor * p=new Tensor(functional::add(x,y));
	// cout<<*p<<endl;

	// cout<<functional::add(y,x)<<endl;
	// cout<<functional::sub(x,y)<<endl;
	// cout<<functional::sub(y,x)<<endl;
	// cout<<functional::add(functional::add(x,y),y)<<endl;

	// cout<<functional::max(x,y)<<endl;
	// cout<<functional::min(x,y)<<endl;

	// Tensor x({2,2}),y({2,2});x={1,2,2,1};y={1,0,1,1};
	// cout<<x<<endl;cout<<y<<endl;
	// cout<<functional::matmul(x,y)<<endl;
	// cout<<functional::matmul_T(x,y)<<endl;
	// cout<<functional::T_matmul(x,y)<<endl;
	// // 1 2  1 0   =   3 2 
	// // 2 1  1 1   =   3 1


	// // Build Graph


	Variable x(3),y(2);
	Variable & w=Graph::Add(x,y);
	Variable & z=Graph::Add(w,w);

	// Don't write Variable z=Graph.Add(x,y); the move constructor will ruin the variable stored in Graph !

	// Graph builds graph first
	cout<<z<<endl;
	
	Graph::eval(z);
	cout<<z<<endl;

	//Auto_diff
	Graph::backward(z);
	cout<<*x.grad<<endl;
	cout<<*y.grad<<endl;

	// Example 2 Gradient Descend
	// Graph g;
	// float lr=0.1;
	// int total=200;

	// Tensor x({2,1}),y({2,1});x={1,1};y={1,2};
	// Variable vx(std::move(x)), vy(std::move(y));

	// Variable M({2,2});*M.data={1,0,2,1};
	// Variable & mx = g.MatMul(M,vx);
	// Variable & residual = g.Sub(mx,vy);
	// Variable & loss = g.InnerProduct(residual,residual);

	// for (int epoch=0;epoch<total;++epoch)
	// {
	// 	g.eval(loss);
	// 	cout<<loss.data->p[0]<<endl;
	// 	g.zero_grad();
	// 	g.backward(loss);
	// 	*vx.data-=functional::cmul(lr,*vx.grad);
	// 	//lr*=0.9;
	// }

	// Graph calculates only the expression that leads to it

	// Graph g;
	// Variable x(2),y(3),z(5);
	// Variable & w=g.Add(x,y);
	// Variable & t=g.Sub(y,z);
	// Variable & t2=g.MatMul(t,x);
	// g.eval(t2);
	// cout<<w<<endl;	// w will be empty because it doesn't lead to t2.
	// cout<<t<<endl;
	// cout<<t2<<endl;

	return 0;
}