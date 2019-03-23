#ifndef DMAP_H
#define DMAP_H

#include "variable.h"
#include "operator.h"
#include <vector>
#include <queue>
#include <algorithm>

namespace dlframework{

class Graph{
public:
	static std::vector<baseOp*> op_stack; //temporary operators; graph nodes
	static std::vector<Variable*> v_stack; //temporary variables; intermediate results of calculation.
	//Graph();
	static void zero_grad();
	static void eval(const Variable &);
	static void backward(Variable &);
	static void clear();
	
	//Capital letter
	static Variable & Add(Variable &, Variable &); //allocate memory; build graph 
	static Variable & Sub(Variable &, Variable &);
	static Variable & MatMul(Variable &, Variable &);
	static Variable & InnerProduct(Variable & , Variable &);

	//virtual ~Graph();
};




}

#endif