/*
C++ version of ISBF for TraceBounds
*/
#include <iostream>
#include <math.h>
#include "isbfcpp.h"

using namespace std;

isbfcpp::isbfcpp()
{
    head = NULL;
    curr = NULL;
}

int isbfcpp::length()
{
  node *cur = head;
  int len = 0;
	while(cur) {
		cur = cur->next;
    len++;
	}
  return len;
}

void isbfcpp::nth_from_last(int n, int &x, int &y)
{
    node *cur = head;
    int count = 0;
  	while(cur) {
        if (count == n) {
            x = cur->val_x;
            y = cur->val_y;
        }
    		cur = cur->next;
        count++;
  	}
}

bool isbfcpp::addList(int x, int y)
{
  node *newNode = new node;
  newNode->val_x = x;
  newNode->val_y = y;
  newNode->next = NULL;

  if (head == NULL) {
    head = new node;
    curr = new node;
    head = curr = newNode;
  }
  else {
    curr-> next = newNode;
    curr = newNode;
  }

  count++;
  return true;
}

void isbfcpp::roateMatrix(int rows, int cols, int **input, int **output)
{
    int i, j;
    for (i=0; i<rows; i++){
      for (j=0;j<cols; j++){
        output[j][rows-1-i] = input[i][j];
      }
    }
}

void isbfcpp::clean()
{
    for (int i = 0; i < nrows; i++) {
        delete[] matrix00[i];
        delete[] matrix180[i];
    }
    for (int i = 0; i < ncols; i++) {
        delete[] matrix90[i];
        delete[] matrix270[i];
    }
    delete[] matrix00;
    delete[] matrix90;
    delete[] matrix180;
    delete[] matrix270;
}

vector<int> isbfcpp::getList(int rows, int cols, int *size, int *mask, int startX, int startY, float inf)
{
    addList(startX, startY);

    nrows = rows;
    ncols = cols;

    matrix00 = new int*[nrows];
    for (int i = 0; i < nrows; i++) {
      matrix00[i] = new int[ncols];
    }

    // copy mask to matrix00
    for(int i=0; i< nrows; i++){
        for(int j=0; j< ncols; j++){
            matrix00[i][j] = mask[i*ncols+j];
        }
    }

    matrix90 = new int*[ncols];
    for (int i = 0; i < ncols; i++) {
      matrix90[i] = new int[nrows];
    }
    matrix180 = new int*[nrows];
    for (int i = 0; i < nrows; i++) {
      matrix180[i] = new int[ncols];
    }
    matrix270 = new int*[ncols];
    for (int i = 0; i < ncols; i++) {
      matrix270[i] = new int[nrows];
    }

    // rotate Matrix for 90, 180, 270 degrees
    roateMatrix(nrows, ncols, matrix00, matrix270);
    roateMatrix(ncols, nrows, matrix270, matrix180);
    roateMatrix(nrows, ncols, matrix180, matrix90);

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

      int x;
      int y;

      x = curr->val_x;
      y = curr->val_y;

      int **h = new int*[rowISBF];
      for (int i = 0; i < rowISBF; i++) {
        h[i] = new int[colISBF];
      }

      // initialize a and b which are indices of ISBF
      int a = 0;
      int b = 0;

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

      // initialize cX and cY
      // cX and cY indicate directions for each ISBF
      vector<int> cX(1);
      vector<int> cY(1);
      cX.at(0)=0;
      cY.at(0)=0;

      if (h[1][0] == 1) {
          // 'left' neighbor
          cX.at(0) = -1;
          cY.at(0) = 0;
          DX = -1;
          DY = 0;
      }
      else{
          if((h[2][0] == 1)&&(h[2][1] != 1)){
              // inner-outer corner at left-rear
              cX.at(0) = -1;
              cY.at(0) = 1;
              DX = 0;
              DY = 1;
          }
          else{
              if(h[0][0] == 1){
                  if(h[0][1] == 1){
                      // inner corner at front
                      cX.at(0) = 0;
                      cY.at(0) = -1;
                      cX.resize(2,(int)-1);
                      cY.resize(2,(int)0);
                      DX = 0;
                      DY = -1;
                  }
                  else{
                      // inner-outer corner at front-left
                      cX.at(0) = -1;
                      cY.at(0) = -1;
                      DX = 0;
                      DY = 1;
                  }
              }
              else if(h[0][1] == 1){
                // front neighbor
                cX.at(0) = 0;
                cY.at(0) = -1;
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

      // free ISBF matrix
      for (int i = 0; i < rowISBF; i++) {
          delete[] h[i];
      }
      delete[] h;

      // transform points by incoming directions and add to contours
      float s = sin(angle);
      float c = cos(angle);

      if(!((cX.at(0)==0)&&(cY.at(0)==0))){
          for(int t=0; t<int(cX.size()); t++){

              float a, b;
              int cx = cX.at(t);
              int cy = cY.at(t);

              a = c*cx - s*cy;
              b = s*cx + c*cy;
              x = curr->val_x;
              y = curr->val_y;

              addList(x+roundf(a), y+roundf(b));
          }
      }

      float i, j;
      i = c*DX-s*DY;
      j = s*DX+c*DY;
      DX = roundf(i);
      DY = roundf(j);

      // get length of the current linked list
      sizeofX = length();

      if (sizeofX > 3) {
          int fx1 = head->val_x;
          int fx2 = head->next->val_x;
          int fy1 = head->val_y;
          int fy2 = head->next->val_y;
          int lx1 = curr->val_x;
          int ly1 = curr->val_y;
          int lx2, ly2;
          nth_from_last(sizeofX-2, lx2, ly2);
          // check if the first and the last x and y are equal
          if ((sizeofX > inf)|| \
            ((lx1 == fx2)&&(lx2 == fx1)&&(ly1 == fy2)&&(ly2 == fy1))){
                // delete the last node
                node *c = head;
                node *temp =  head;
                while(c->next != NULL){
                  temp = c;
                  c = c->next;
                }
                temp->next = NULL;

                break;
          }
      }
    }
    // allocate memory for return value
    vector<int> array(2*sizeofX-2);
    int index = 0;
    while (head != NULL) {
        array[index] = head->val_x;
        array[index+sizeofX-1] = head->val_y;
        head = head->next;
        index++;
    }

    // get size of x
    *size = sizeofX;

    clean();

    return array;
}

isbfcpp::~isbfcpp()
{
    node *c = head;
    while (c != NULL) {
        head = head->next;
        delete c;
        c = head;
    }
    curr = head = NULL;
}
