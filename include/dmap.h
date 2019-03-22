// #ifndef DMAP_H
// #define DMAP_H

// #include "variable.h"
// #include "operator.h"
// #include <stack>

// namespace dlframework{

// class Graph{
// public:
// 	std::vector<baseOp*> op_stack; //temporary operators; dynamic graph nodes
// 	std::vector<Variable*> v_stack; //temporary variables; intermediate results of calculation.
// 	Graph();
// 	void backward(const Variable &);
// 	void clear();
	
// 	//Capital letter
// 	Variable & Add(const Variable &, const Variable &); //allocate memory; build graph 

// 	virtual ~Graph();
// };




// }

// #endif