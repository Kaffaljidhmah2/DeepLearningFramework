#ifndef DMAP_H
#define DMAP_H

#include "variable.h"
#include "operator.h"
#include <vector>

namespace dlframework{

class Graph{
public:
	std::vector<baseOp*> op_stack; //temporary operators; graph nodes
	std::vector<Variable*> v_stack; //temporary variables; intermediate results of calculation.
	Graph();
	void zero_grad();
	void eval(const Variable &);
	void backward(Variable &);
	void clear();
	
	//Capital letter
	Variable & Add(Variable &, Variable &); //allocate memory; build graph 
	Variable & Sub(Variable &, Variable &);
	Variable & MatMul(Variable &, Variable &);
	Variable & InnerProduct(Variable & , Variable &);

	virtual ~Graph();
};




}

#endif