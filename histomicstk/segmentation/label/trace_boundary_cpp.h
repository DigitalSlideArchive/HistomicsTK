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
    void rot90(int nrows, int ncols,
               std::vector <std::vector<int> > input,
               std::vector <std::vector<int> > &output);
    std::vector <std::vector<std::vector<int>> > trace_boundary(std::vector <std::vector<int> > imLables, int connectivity);
    std::vector <std::vector<int> > isbf(int nrows, int ncols, std::vector <std::vector<int> > mask,
      std::vector <std::vector<int> > mask_90, std::vector <std::vector<int> > mask_180,
      std::vector <std::vector<int> > mask_270, int startX, int startY, float inf);
    std::vector <std::vector<int> > moore(int nrows, int ncols, std::vector <std::vector<int> > mask,
      std::vector <std::vector<int> > mask_90, std::vector <std::vector<int> > mask_180,
      std::vector <std::vector<int> > mask_270, int startX, int startY, float inf);
    ~trace_boundary_cpp();

};

#endif
