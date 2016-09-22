/*
C++ version of ISBF for TraceBounds
*/
#ifndef isbfcpp_h
#define isbfcpp_h

#include <iostream>
#include <vector>

class isbfcpp
{

public:
    isbfcpp();
    std::vector <std::vector<int> > rotateMatrix(int rows, int cols,  std::vector <std::vector<int> > input);
    std::vector <std::vector<int> > traceBoundary(int nrows, int ncols, std::vector <std::vector<int> > mask, int startX, int startY, float inf);
    ~isbfcpp();
};

#endif
