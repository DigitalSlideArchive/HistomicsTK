/*
C++ version of ISBF for TraceBounds
*/
#ifndef isbfcpp_h
#define isbfcpp_h

#include <iostream>
#include <string>
#include <vector>

struct node
{
    int val_x;
    int val_y;
    node *next;
};

class isbfcpp
{
private:
    node *head;
    node *curr;
    int count;
    int nrows;
    int ncols;
    int** matrix00;
    int** matrix90;
    int** matrix180;
    int** matrix270;

public:
    isbfcpp();
    int length();
    void nth_from_last(int n, int &x, int &y);
    bool addList(int x, int y);
    void rotateMatrix(int rows, int cols, int **input, int **output);
    std::vector<int> getList(int nrows, int ncols, int *size, int *mask, int startX, int startY, float inf);
    void clean();
    ~isbfcpp();
};

#endif
