/*
C++ version of Moore for TraceBounds
*/
#ifndef trace_boundary_cpp_h
#define trace_boundary_cpp_h

#include <iostream>
#include <vector>

#ifndef INFINITY
#define INFINITY 0
#endif

class trace_boundary_cpp
{

public:
    trace_boundary_cpp();
    void rot90(int nrows, int ncols, std::vector <std::vector<int> > matrix,
               std::vector <std::vector<int> > &matrix270,
               std::vector <std::vector<int> > &matrix180,
               std::vector <std::vector<int> > &matrix90);
    std::vector <std::vector<int> > isbf(int nrows, int ncols, std::vector <std::vector<int> > mask, int startX, int startY, float inf);
    std::vector <std::vector<int> > moore(int nrows, int ncols, std::vector <std::vector<int> > mask, int startX, int startY, float inf);
    ~trace_boundary_cpp();

};

#endif
