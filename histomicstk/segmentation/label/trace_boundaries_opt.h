/*
Header of trace_boundaries_opt
*/
#ifndef trace_boundaries_opt_hpp
#define trace_boundaries_opt_hpp

#include <vector>

#ifndef INFINITY
#define INFINITY 0
#endif

void rot90(int nrows, int ncols, std::vector <std::vector<int> > input, std::vector <std::vector<int> > &output);
std::vector <std::vector<std::vector<int>> > trace_boundary(std::vector <std::vector<int> > imLables, int connectivity=4,
    float inf=INFINITY, int startX=-1, int startY=-1);
std::vector <std::vector<std::vector<int>> > trace_label(std::vector <std::vector<int> > imLables, int connectivity=4,
    float inf=INFINITY);
std::vector <std::vector<int> > isbf(int nrows, int ncols, std::vector <std::vector<int> > mask,
    std::vector <std::vector<int> > mask_90, std::vector <std::vector<int> > mask_180,
    std::vector <std::vector<int> > mask_270, int startX, int startY, float inf);
std::vector <std::vector<int> > moore(int nrows, int ncols, std::vector <std::vector<int> > mask,
    std::vector <std::vector<int> > mask_90, std::vector <std::vector<int> > mask_180,
    std::vector <std::vector<int> > mask_270, int startX, int startY, float inf);

#endif
