/*
C++ version of ISBF for TraceBounds
*/
#include "isbfcpp.h"

#include <iostream>
#include <list>
#include <cmath>
#include <vector>

isbfcpp::isbfcpp(){}

std::vector <std::vector<int> > isbfcpp::rotateMatrix(int nrows, int ncols, std::vector <std::vector<int> > input)
{
    std::vector <std::vector<int> > output(ncols, std::vector<int>(nrows));
    for (int i=0; i<nrows; i++){
      for (int j=0;j<ncols; j++){
        output[j][nrows-1-i] = input[i][j];
      }
    }
    return output;
}

std::vector <std::vector<int> > isbfcpp::traceBoundary(int nrows, int ncols, std::vector <std::vector<int> > mask, int startX, int startY, float inf)
{
    // initialize boundary vector
    std::list<int> boundary_listX;
    std::list<int> boundary_listY;

    // push the first x and y points
    boundary_listX.push_back(startX);
    boundary_listY.push_back(startY);

    // initialize matrix for 0, 90, 180, 270 degrees
    std::vector <std::vector<int> > matrix00(nrows, std::vector<int>(ncols));
    std::vector <std::vector<int> > matrix90(ncols, std::vector<int>(nrows));
    std::vector <std::vector<int> > matrix180(nrows, std::vector<int>(ncols));
    std::vector <std::vector<int> > matrix270(ncols, std::vector<int>(nrows));

    // copy mask to matrix00
    for(int i=0; i< nrows; i++){
        for(int j=0; j< ncols; j++){
            //matrix00[i][j] = mask[i*ncols+j];
            matrix00[i][j] = mask[i][j];
        }
    }

    // rotate matrix for 90, 180, 270 degrees
    matrix270 = rotateMatrix(nrows, ncols, matrix00);
    matrix180 = rotateMatrix(ncols, nrows, matrix270);
    matrix90 = rotateMatrix(nrows, ncols, matrix180);

    // set defalut direction
    int DX = 1;
    int DY = 0;

    // set the number of rows and cols for ISBF
    int rowISBF = 3;
    int colISBF = 2;

    float angle;

    // set size of X: the size of X is equal to the size of Y
    int sizeofX;

    // loop until true
    while(1) {

      std::vector <std::vector<int> > h(rowISBF, std::vector<int>(colISBF));

      // initialize a and b which are indices of ISBF
      int a = 0;
      int b = 0;

      int x = boundary_listX.back();
      int y = boundary_listY.back();

      //cout << DX << " " << DY << endl;

      if((DX == 1)&&(DY == 0)){
        for (int i = ncols-x-2; i < ncols-x+1; i++) {
          for (int j = y-1; j < y+1; j++) {
              h[a][b] = matrix90[i][j];
              b++;
          }
          b = 0;
          a++;
        }
        angle = M_PI/2;
      }

      else if((DX == 0)&&(DY == -1)){
        for (int i = y-1; i < y+2; i++) {
          for (int j = x-1; j < x+1; j++) {
              h[a][b] = matrix00[i][j];
              b++;
          }
          b = 0;
          a++;
        }
        angle = 0;
      }

      else if((DX == -1)&&(DY == 0)){
        for (int i = x-1; i < x+2; i++) {
          for (int j = nrows-y-2; j < nrows-y; j++) {
              h[a][b] = matrix270[i][j];
              b++;
          }
          b = 0;
          a++;
        }
        angle = 3*M_PI/2;
      }

      else{
        for (int i = nrows-y-2; i < nrows-y+1; i++) {
          for (int j = ncols-x-2; j < ncols-x; j++) {
              h[a][b] = matrix180[i][j];
              b++;
          }
          b = 0;
          a++;
        }
        angle = M_PI;
      }

      // initialize cX and cY which indicate directions for each ISBF
      std::vector<int> cX(1);
      std::vector<int> cY(1);

      if (h[1][0] == 1) {
          // 'left' neighbor
          cX[0] = -1;
          cY[0] = 0;
          DX = -1;
          DY = 0;
      }
      else{
          if((h[2][0] == 1)&&(h[2][1] != 1)){
              // inner-outer corner at left-rear
              cX[0] = -1;
              cY[0] = 1;
              DX = 0;
              DY = 1;
          }
          else{
              if(h[0][0] == 1){
                  if(h[0][1] == 1){
                      // inner corner at front
                      cX[0] = 0;
                      cY[0] = -1;
                      cX.push_back(-1);
                      cY.push_back(0);
                      DX = 0;
                      DY = -1;
                  }
                  else{
                      // inner-outer corner at front-left
                      cX[0] = -1;
                      cY[0] = -1;
                      DX = 0;
                      DY = -1;
                  }
              }
              else if(h[0][1] == 1){
                // front neighbor
                cX[0] = 0;
                cY[0] = -1;
                DX = 1;
                DY = 0;
              }
              else{
                // outer corner
                DX = 0;
                DY = 1;
              }
          }
      }

      // transform points by incoming directions and add to contours
      float s = sin(angle);
      float c = cos(angle);

      if(!((cX[0]==0)&&(cY[0]==0))){
          for(int t=0; t< int(cX.size()); t++){
              float a, b;
              int cx = cX[t];
              int cy = cY[t];

              a = c*cx - s*cy;
              b = s*cx + c*cy;

              x = boundary_listX.back();
              y = boundary_listY.back();

              boundary_listX.push_back(x+roundf(a));
              boundary_listY.push_back(y+roundf(b));
          }
      }

      float i, j;
      i = c*DX - s*DY;
      j = s*DX + c*DY;
      DX = roundf(i);
      DY = roundf(j);

      // get length of the current linked list
      sizeofX = boundary_listX.size();

      if (sizeofX > 3) {

          // check if the first and the last x and y are equal
          if ((sizeofX > inf)|| \
            ((*std::prev(boundary_listX.end()) == *std::next(boundary_listX.begin(), 1))&&
            (*std::prev(boundary_listX.end(), 2) == *boundary_listX.begin())&&
            (*std::prev(boundary_listY.end()) == *std::next(boundary_listY.begin(), 1))&&
            (*std::prev(boundary_listY.end(), 2) == *boundary_listY.begin()))){
              break;
          }
      }
    }

    // allocate memory for return value
    std::vector <std::vector<int> > boundaries(2, std::vector<int>(sizeofX));

    // remove the last element
    boundary_listX.pop_back();
    boundary_listY.pop_back();
    /*
    std::vector<int> bX { std::make_move_iterator(begin(boundary_listX)),
                     std::make_move_iterator(end(boundary_listX)) };

    std::vector<int> bY { std::make_move_iterator(begin(boundary_listY)),
                    std::make_move_iterator(end(boundary_listY)) };
    */
    std::vector<int> bX (boundary_listX.begin(), boundary_listX.end());

    std::vector<int> bY (boundary_listY.begin(), boundary_listY.end());

    boundaries[0] = bX;
    boundaries[1] = bY;

    return boundaries;
}

isbfcpp::~isbfcpp(){}
