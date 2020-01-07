#include "variable.h"
#include "operator.h"
#include "functional.h"
#include "dmap.h"
#include "nn.h"
#include "optim.h"
#include "utils.h"
#include <iostream>
#include <chrono>

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
	// Tensor z(x,true); /* z = zero_like (x) */
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
	// Variable y(1.2,false);cout<<y<<endl;
	// Variable z({2,3,4});cout<<z<<endl;

	//Shadow Copy
	// Variable y(1.2,false);cout<<y<<endl;
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
	// cout<<functional::add(x,y)<<endl;
	// Tensor * p=new Tensor(functional::add(x,y));
	// cout<<*p<<endl;

	// cout<<functional::add(y,x)<<endl;
	// cout<<functional::sub(x,y)<<endl;
	// cout<<functional::sub(y,x)<<endl;
	// cout<<functional::add(functional::add(x,y),y)<<endl;

	// cout<<functional::max(x,y)<<endl;
	// cout<<functional::min(x,y)<<endl;

	// mat mul

	// Tensor x({2,2}),y({2,2});x={1,2,2,1};y={1,0,1,1};
	// cout<<x<<endl;cout<<y<<endl;
	// cout<<functional::matmul(x,y)<<endl;
	// cout<<functional::matmul_T(x,y)<<endl;
	// cout<<functional::T_matmul(x,y)<<endl;

	// // 1 2  1 0   =   3 2 
	// // 2 1  1 1   =   3 1


	// //Softmax
	// Tensor x({2,2});x={0,0,0,0};
	// cout<<functional::softmax(x)<<endl;

	


	// Random number test
	// Init::set_seed(std::chrono::system_clock::now().time_since_epoch().count());

	// Tensor x({20000}); Init::normal(x);
	// cout<<functional::mean(x)<<endl;
	// cout<<functional::mean(functional::e_mul(x,x))<<endl;


	// // Build Graph


	// Variable x(3,true),y(2,true); //requires_grad=true
	// Variable & w=Graph::Add(x,y);
	// Variable & z=Graph::Add(x,w);

	// // Don't write "Variable z=Graph::Add(x,y);" the move constructor will ruin the variable stored in Graph !

	// // Graph builds graph first
	// cout<<z<<endl;
	
	// // Graph::eval computes the calculation that leads to the variable.
	
	// Graph::eval(z);
	// cout<<z<<endl;

	// //Auto_diff
	// Graph::backward(z);
	// cout<<*x.grad<<endl;
	// cout<<*y.grad<<endl;

	// // Example 2 Gradient Descend
	// float lr=0.1;
	// int total=200;

	// Tensor x({2,1}),y({2,1});x={1,1};y={1,2};
	// Variable vx(std::move(x),true), vy(std::move(y));

	// Variable M({2,2});*M.data={1,0,2,1};
	// Variable & mx = Graph::MatMul(M,vx);
	// Variable & residual = Graph::Sub(mx,vy);
	// Variable & loss = Graph::InnerProduct(residual,residual);

	// for (int epoch=0;epoch<total;++epoch)
	// {
	// 	Graph::eval(loss);
	// 	cout<<loss.data->p[0]<<endl;
	// 	Graph::zero_grad();
	// 	Graph::backward(loss);
	// 	*vx.data-=functional::cmul(lr,*vx.grad);
	// 	//lr*=0.9;
	// }

	// Graph calculates only the expression that leads to it
	
	// Variable x(2),y(3),z(5);
	// Variable & w=Graph::Add(x,y);
	// Variable & t=Graph::Sub(y,z);
	// Variable & t2=Graph::MatMul(t,x);
	// Graph::eval(t2);
	// cout<<w<<endl;	// w will be empty because it doesn't lead to t2.
	// cout<<t<<endl;
	// cout<<t2<<endl;

	// // Test for ReLU module
	// Variable M({2,2},true);*M.data={-1,-2,-1,1};
	// Variable & out=Graph::ReLU(M);
	// Variable & l=Graph::InnerProduct(out,out);
	
	// Graph::eval(l);
	// Graph::zero_grad();
	// Graph::backward(l);
	// cout<<*M.grad<<endl;

	// // Test for Linear module

	// Variable x({3,1});*x.data={20,0.2,-0.3};
	// nn::Linear fc(3,2);
	// *fc.weight.data={1,2,3,4,5,10};
	// *fc.bias.data={0,2,3};
	// Variable y({2,1});*y.data={1,-1,0};
	// Variable & out=fc(x);
	// Variable & residual=Graph::Sub(out,y);
	// Variable & loss=Graph::InnerProduct(residual,residual);

	// float lr=0.001;
	// for (int epoch=0;epoch<20;++epoch)
	// {
	// 	Graph::eval(loss);
	// 	Graph::zero_grad();
	// 	Graph::backward(loss);
	// 	// cout<<Graph::v_stack.size()<<endl;
	// 	// cout<<Graph::op_stack.size()<<endl;
	// 	// cout<<fc.weight<<endl;
	// 	// cout<<fc.bias<<endl;
	// 	// cout<<out<<endl;
	// 	cout<<loss.data->p[0]<<endl;

	// 	//cout<<*fc.weight.grad<<endl;
	// 	*fc.weight.data-=functional::cmul(lr,*fc.weight.grad);
	// 	*fc.bias.data-=functional::cmul(lr,*fc.bias.grad);	
	// }
	
	// // Define your module !

	// class MyNet{
	// public:
	// 	nn::Linear fc1;
	// 	nn::Linear fc2;
	// 	MyNet():fc1(3,5),fc2(5,1){}

	// 	Variable & operator()(Variable & x)
	// 	{
	// 		Variable & out1=fc1(x);
	// 		Variable & out2=Graph::ReLU(out1);
	// 		Variable & out3=fc2(out2);
	// 		return out3;
	// 	}
	// };

	// MyNet mynet;

	// cout<<Optimizer::Params.size()<<endl;
	// SGDOptimizer myoptim(2, 0.1, 5e-4);    //SGDOptimizer(int bs, float lr=0.1, float wd=0);

	// cout<<"Before mynet() "<<Graph::op_stack.size()<<endl;
	// Variable input({3,1}); input={1,2,5}; Variable target({2,1}); target={1,-4}; //Data sets
	// Variable & out = mynet(input);
	// cout<<"After mynet() "<<Graph::op_stack.size()<<endl;	

	// Variable & residual = Graph::Sub(out, target);
	// Variable & loss = Graph::InnerProduct(residual, residual);

	// cout<<"After loss defined "<<Graph::op_stack.size()<<endl;	
	// cout<<Optimizer::Params.size()<<endl;
	// for (int idx=0;idx<200;++idx)	
	// {
	// 	cout<<"idx: "<<idx<<endl;	
	// 	Graph::eval(loss);
	// 	cout<<loss.data->p[0]<<endl;
	// 	Graph::zero_grad();
	// 	Graph::backward(loss);
	// 	myoptim.step();
	// }

	// // Test MNIST dataset.
	// Tensor ** train_image=dataset::Read_MNIST_Train_Image("../dataset/train-images-idx3-ubyte");
	// Tensor ** train_label=dataset::Read_MNIST_Train_Label("../dataset/train-labels-idx1-ubyte");
	// Tensor ** test_image=dataset::Read_MNIST_Test_Image("../dataset/t10k-images-idx3-ubyte");
	// Tensor ** test_label=dataset::Read_MNIST_Test_Label("../dataset/t10k-labels-idx1-ubyte");

	// // //Visualize images.
	// // dataset::Visualize_Grayscale(*train_image[59999],cout);
	// // cout<<train_label[59999]->p[0]<<endl;

	// //initalize random seed
	// Init::set_seed(std::chrono::system_clock::now().time_since_epoch().count());

	// // Define a 2-layer NN.
	// class MyNet{
	// public:
	// 	nn::Linear fc1;
	// 	nn::Linear fc2;
	// 	MyNet():fc1(28*28,500),fc2(500,10){}

	// 	Variable & operator()(Variable & x)
	// 	{
	// 		Variable & out1=fc1(x);
	// 		Variable & out2=Graph::ReLU(out1);
	// 		Variable & out3=fc2(out2);
	// 		return out3;
	// 	}
	// };

	// MyNet mynet;

	// Variable x,y; // Use default constructor, otherwise you need to clear_data to release the initial tensor.

	// //Reshape train and test images

	// for (int i=0;i<60000;++i)
	// {
	// 	train_image[i]->reshape({28*28,1});
	// }
	// for (int j=0;j<10000;++j)
	// {
	// 	test_image[j]->reshape({28*28,1});
	// }

	// //Build Graph

	// Variable & out = mynet(x);
	// Variable & loss = Graph::SoftmaxCrossEntropy(out,y);

	// // Define an optimizer
	// int batch_size=100;
	// float learning_rate=0.1;
	// float weight_decay=5e-4;
	// SGDOptimizer myoptim(batch_size, learning_rate, weight_decay);


	// int ran_indices[60000];
	// for (int i=0;i<60000;++i) ran_indices[i]=i;
	// for (int epoch=0;epoch<1;++epoch)	
	// {
	// 	cout<<"epoch "<<epoch<<endl;
	// 	random_shuffle(ran_indices,ran_indices+60000);
	// 	for (int inner_loop=0; inner_loop<60000; ++inner_loop)
	// 	{
	// 		//Manual Assign tensor to x and y.
	// 		x.data=train_image[ran_indices[inner_loop]];
	// 		y.data=train_label[ran_indices[inner_loop]];
	// 		Graph::eval(loss);
	// 		if ((inner_loop+1) % batch_size ==0)
	// 			cout<<loss.data->p[0]<<endl;
	// 		Graph::zero_grad();
	// 		Graph::backward(loss);
	// 		myoptim.step();
	// 	}
	// }

	// //Eval on Test

	// int total_correct=0;
	// for (int inner_loop=0; inner_loop<10000; ++ inner_loop)
	// {
	// 	x.data=test_image[inner_loop];
	// 	Graph::eval(out);
	// 	Tensor u=functional::softmax(*out.data);
	// 	auto argmax = std::max_element(u.p,u.p+u.length);
	// 	if ((argmax - u.p) == (int)(test_label[inner_loop]->p[0]))
	// 		++total_correct;
	// }
	// cout<<"Numbers of correctness: "<<total_correct<<endl;

	

	// // Free the resources.
	// x.data=nullptr;
	// y.data=nullptr;

	// for (int i=0;i<60000;++i)
	// {
	// 	delete train_image[i];
	// 	delete train_label[i];
	// }
	// delete train_image;
	// delete train_label;

	// for (int j=0;j<10000;++j)
	// {
	// 	delete test_image[j];
	// 	delete test_label[j];
	// }	
	// delete test_image;
	// delete test_label;

	// Test for conv

	// Tensor x({2,4,3}); 
	// x={1,2,3,
	// 	2,3,4,
	// 	1,4,2};
	// Tensor k({1,2,3,3});
	// k={0,0,0,
	// 	-1,1,0,
	// 	0,0,0};
	// cout<<functional::conv3d(x,k,1,1)<<endl;


	return 0;
}