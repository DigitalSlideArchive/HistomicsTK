/*
Source of trace_boundaries_opt
*/
#include "trace_boundaries_opt.h"

#include <iostream>
#include <list>
#include <cmath>
#include <vector>
#include <set>
#include <algorithm>

struct Points {
    int x, y;
};

void rot90(
  std::vector <std::vector<int> > input,
  std::vector <std::vector<int> > &output)
{
    int nrows = input.size();
    int ncols = input[0].size();

    for(int i = 0; i < nrows; i++){
      for (int j = 0; j < ncols; j++){
        output[j][nrows-1-i] = input[i][j];
      }
    }
}

std::vector <std::vector<std::vector<int> > > trace_boundary(
    std::vector <std::vector<int> > imLabel,
    int connectivity,
    int startX, int startY, float max_length)
{
    std::vector <std::vector<std::vector<int> > > output;

    int nrows = imLabel.size();
    int ncols = imLabel[0].size();

    // compute rotated versions of label mask
    std::vector<std::vector<int> > imLabel_90(ncols, std::vector<int>(nrows));
    std::vector<std::vector<int> > imLabel_180(nrows, std::vector<int>(ncols));
    std::vector<std::vector<int> > imLabel_270(ncols, std::vector<int>(nrows));

    rot90(imLabel, imLabel_270);
    rot90(imLabel_270, imLabel_180);
    rot90(imLabel_180, imLabel_90);

    // find starting x and y points if not defined
    if ((startX == -1)&&(startY == -1)) {

        bool flag = false;

        for(int i = 1; i < nrows-1; i++) {

          for(int j = 1; j < ncols-1; j++) {

             if(imLabel[i][j] > 0 && !flag) {

               // check if the number of points is one
               if(!(imLabel[i][j+1] == 0 && imLabel[i+1][j] == 0 &&
                    imLabel[i+1][j+1] == 0 && imLabel[i-1][j+1] == 0)) {
                   startX = j;
                   startY = i;
                   flag = true;
                   break;
               }
             }
          }

          if(flag)
            break;
        }
    }

    std::vector <std::vector<int> > coords;

    if (connectivity == 4) {
        coords = isbf(
            imLabel, imLabel_90, imLabel_180, imLabel_270,
            startX, startY, max_length);
    }
    else {
        coords = moore(
            imLabel, imLabel_90, imLabel_180, imLabel_270,
            startX, startY, max_length);
    }

    // append current coords to output vector
    output.push_back(coords);

    return output;
}

std::vector <std::vector<std::vector<int> > > trace_label(
    std::vector <std::vector<int> > imLabel,
    int connectivity,
    float max_length)
{
    std::vector <std::vector<std::vector<int> > > output;

    int nrows = imLabel.size();
    int ncols = imLabel[0].size();

    // compute rotated versions of label mask
    std::vector<std::vector<int> > imLabel_90(ncols, std::vector<int>(nrows));
    std::vector<std::vector<int> > imLabel_180(nrows, std::vector<int>(ncols));
    std::vector<std::vector<int> > imLabel_270(ncols, std::vector<int>(nrows));

    rot90(imLabel, imLabel_270);
    rot90(imLabel_270, imLabel_180);
    rot90(imLabel_180, imLabel_90);

    // find label ids of objects present
    std::set<int> label_set;

    std::vector <std::vector<int> >::iterator row;
    std::vector<int>::iterator col;

    for(row = imLabel.begin(); row != imLabel.end(); row++) {
        for(col = row->begin(); col != row->end(); col++) {
          if(*col != 0){
            label_set.insert(*col);
          }
        }
    }

    // trace each object one at a time
    std::set<int>::iterator it;

    for (it = label_set.begin(); it != label_set.end(); it++) {

        int cur_lid = *it;

        // find x and y bounds of current object
        std::vector<Points> point;

        for (int i=0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                if(imLabel[i][j] == cur_lid) {
                    point.push_back({i, j});
                }
            }
        }

        auto mmX = std::minmax_element(point.begin(), point.end(),
            [] (Points const& lhs, Points const& rhs) {return lhs.x < rhs.x;});
        auto mmY = std::minmax_element(point.begin(), point.end(),
            [] (Points const& lhs, Points const& rhs) {return lhs.y < rhs.y;});

        int minX = mmX.first->x;
        int maxX = mmX.second->x;
        int minY = mmY.first->y;
        int maxY = mmY.second->y;

        // initialize number of rows and cols of mask with padding
        int nrows_mask = maxX - minX + 3;
        int ncols_mask = maxY - minY + 3;

        std::vector<std::vector<int> > mask(
            nrows_mask, std::vector<int>(ncols_mask, 0));

        std::vector<std::vector<int> > mask_90(
            ncols_mask, std::vector<int>(nrows_mask, 0));

        std::vector<std::vector<int> > mask_180(
            nrows_mask, std::vector<int>(ncols_mask, 0));

        std::vector<std::vector<int> > mask_270(
            ncols_mask, std::vector<int>(nrows_mask, 0));

        for (int i = 1; i < nrows_mask-1; i++) {

            for (int j = 1; j < ncols_mask-1; j++) {

                if(imLabel[minX+i-1][minY+j-1] == cur_lid)
                    mask[i][j] = 1;

                if(imLabel_90[ncols-maxY+j-2][minX+i-1] == cur_lid)
                    mask_90[j][i] = 1;

                if(imLabel_180[nrows-maxX+i-2][ncols-maxY+j-2] == cur_lid)
                    mask_180[i][j] = 1;

                if(imLabel_270[minY+j-1][nrows-maxX+i-2] == cur_lid)
                    mask_270[j][i] = 1;
            }
        }


        // find starting x and y points
        int startX = 0;
        int startY = 0;

        //if ((startX == -1)&&(startY == -1)) {
        bool flag = false;

        for(int i = 1; i < nrows_mask-1; i++) {

          for(int j = 1; j < ncols_mask-1; j++) {

             if(mask[i][j] > 0 && !flag) {

               // check if the nubmer of points is one
               if(!(mask[i][j+1] == 0 && mask[i+1][j] == 0 &&
                    mask[i+1][j+1] == 0 && mask[i-1][j+1] == 0)) {
                   startX = j;
                   startY = i;
                   flag = true;
                   break;
               }

             }
          }

          if(flag)
            break;
        }
        //}
        /*
        else {
            startY = startY - minX + 1;
            startX = startX - minY + 1;
        }
        */
        std::vector <std::vector<int> > coords;

        if (connectivity == 4) {
            coords = isbf(
                mask, mask_90, mask_180, mask_270,
                startX, startY, max_length);
        }
        else {
            coords = moore(
                mask, mask_90, mask_180, mask_270,
                startX, startY, max_length);
        }

        int ncols_coords = coords[0].size();

        // add window offset from original labels coordinates
        for(int i = 0; i < ncols_coords; i++) {
             coords[0][i] = coords[0][i] + minY - 1;
             coords[1][i] = coords[1][i] + minX - 1;
        }

        // append current coords to output vector
        output.push_back(coords);
    }

    return output;
}


std::vector <std::vector<int> > moore(
    std::vector <std::vector<int> > mask,
    std::vector <std::vector<int> > mask_90,
    std::vector <std::vector<int> > mask_180,
    std::vector <std::vector<int> > mask_270,
    int startX, int startY, float max_length)
{
    int nrows = mask.size();
    int ncols = mask[0].size();

    // initialize boundary vector
    std::list<int> boundary_listX;
    std::list<int> boundary_listY;

    // push the first x and y points
    boundary_listX.push_back(startX);
    boundary_listY.push_back(startY);

    // check degenerate case where mask contains 1 pixel
    int sum = 0;
    for(int i=0; i< nrows; i++){
        for(int j=0; j< ncols; j++){
            sum = sum + mask[i][j];
        }
    }

    // set size of X: the size of X is equal to the size of Y
    int sizeofX = 0;

    if (sum > 1) {

      // set defalut direction
      int DX = 1;
      int DY = 0;

      // define clockwise ordered indices
      int row[8] = {2, 1, 0, 0, 0, 1, 2, 2};
      int col[8] = {0, 0, 0, 1, 2, 2, 2, 1};
      int dX[8] = {-1, 0, 0, 1, 1, 0, 0, -1};
      int dY[8] = {0, -1, -1, 0, 0, 1, 1, 0};
      int oX[8] = {-1, -1, -1, 0, 1, 1, 1, 0};
      int oY[8] = {1, 0, -1, -1, -1, 0, 1, 1};

      // set the number of rows and cols for moore
      int rowMoore = 3;
      int colMoore = 3;

      float angle;

      // loop until true
      while(1) {

        std::vector <std::vector<int> > h(rowMoore, std::vector<int>(colMoore));

        // initialize a and b which are indices of moore
        int a = 0;
        int b = 0;
        int x = boundary_listX.back();
        int y = boundary_listY.back();

        if((DX == 1)&&(DY == 0)){
          for (int i = ncols-x-2; i < ncols-x+1; i++) {
            for (int j = y-1; j < y+2; j++) {
                h[a][b] = mask_90[i][j];
                b++;
            }
            b = 0;
            a++;
          }
          angle = M_PI/2;
        }

        else if((DX == 0)&&(DY == -1)){
          for (int i = y-1; i < y+2; i++) {
            for (int j = x-1; j < x+2; j++) {
                h[a][b] = mask[i][j];
                b++;
            }
            b = 0;
            a++;
          }
          angle = 0;
        }

        else if((DX == -1)&&(DY == 0)){
          for (int i = x-1; i < x+2; i++) {
            for (int j = nrows-y-2; j < nrows-y+1; j++) {
                h[a][b] = mask_270[i][j];
                b++;
            }
            b = 0;
            a++;
          }
          angle = 3*M_PI/2;
        }

        else{
          for (int i = nrows-y-2; i < nrows-y+1; i++) {
            for (int j = ncols-x-2; j < ncols-x+1; j++) {
                h[a][b] = mask_180[i][j];
                b++;
            }
            b = 0;
            a++;
          }
          angle = M_PI;
        }

        int Move = 0;
        bool is_moore = false;

        for (int i=0; i<8 ; i++){
          if(!is_moore){
            if (h[row[i]][col[i]] == 1){
                Move = i;
                is_moore = true;
            }
          }
        }
        int cX = oX[Move];
        int cY = oY[Move];
        DX = dX[Move];
        DY = dY[Move];

        // transform points by incoming directions and add to contours
        float s = sin(angle);
        float c = cos(angle);

        float p, q;

        p = c*cX - s*cY;
        q = s*cX + c*cY;

        boundary_listX.push_back(x+roundf(p));
        boundary_listY.push_back(y+roundf(q));

        float i, j;
        i = c*DX-s*DY;
        j = s*DX+c*DY;
        DX = roundf(i);
        DY = roundf(j);

        // get length of the current linked list
        sizeofX = boundary_listX.size();

        if (sizeofX > 3) {

          int fx1 = *boundary_listX.begin();
          int fx2 = *std::next(boundary_listX.begin(), 1);
          int fy1 = *boundary_listY.begin();
          int fy2 = *std::next(boundary_listY.begin(), 1);
          int lx1 = *std::prev(boundary_listX.end());
          int ly1 = *std::prev(boundary_listY.end());
          int lx2 = *std::prev(boundary_listX.end(), 2);
          int ly2 = *std::prev(boundary_listY.end(), 2);

          // check if the first and the last x and y are equal
          if ((sizeofX > max_length)|| \
          ((lx1 == fx2)&&(lx2 == fx1)&&(ly1 == fy2)&&(ly2 == fy1))){
            // remove the last element
              boundary_listX.pop_back();
              boundary_listY.pop_back();
              break;
          }
        }
      }
    }

    std::vector <std::vector<int> > boundary(2, std::vector<int>(sizeofX));

    boundary[0].assign(boundary_listX.begin(), boundary_listX.end());
    boundary[1].assign(boundary_listY.begin(), boundary_listY.end());

    return boundary;
}


std::vector <std::vector<int> > isbf(
    std::vector <std::vector<int> > mask,
    std::vector <std::vector<int> > mask_90,
    std::vector <std::vector<int> > mask_180,
    std::vector <std::vector<int> > mask_270,
    int startX, int startY, float max_length)
{
    int nrows = mask.size();
    int ncols = mask[0].size();

    // initialize boundary vector
    std::list<int> boundary_listX;
    std::list<int> boundary_listY;

    // push the first x and y points
    boundary_listX.push_back(startX);
    boundary_listY.push_back(startY);

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

      if((DX == 1)&&(DY == 0)){
        for (int i = ncols-x-2; i < ncols-x+1; i++) {
          for (int j = y-1; j < y+1; j++) {
              h[a][b] = mask_90[i][j];
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
              h[a][b] = mask[i][j];
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
              h[a][b] = mask_270[i][j];
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
              h[a][b] = mask_180[i][j];
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

        int fx1 = *boundary_listX.begin();
        int fx2 = *std::next(boundary_listX.begin(), 1);
        int fy1 = *boundary_listY.begin();
        int fy2 = *std::next(boundary_listY.begin(), 1);
        int lx1 = *std::prev(boundary_listX.end());
        int ly1 = *std::prev(boundary_listY.end());
        int lx2 = *std::prev(boundary_listX.end(), 2);
        int ly2 = *std::prev(boundary_listY.end(), 2);
        int lx3 = *std::prev(boundary_listX.end(), 3);
        int ly3 = *std::prev(boundary_listY.end(), 3);
        int lx4 = *std::prev(boundary_listX.end(), 4);
        int ly4 = *std::prev(boundary_listY.end(), 4);

        // check if the first and the last x and y are equal
        if ((sizeofX > max_length)|| \
        ((lx1 == fx2)&&(lx2 == fx1)&&(ly1 == fy2)&&(ly2 == fy1))){
          // remove the last element
            boundary_listX.pop_back();
            boundary_listY.pop_back();
            break;
        }
        if (int(cX.size()) == 2)
          if ((lx2 == fx2)&&(lx3 == fx1)&&(ly2 == fy2)&&(ly3 == fy1)){
            boundary_listX.pop_back();
            boundary_listY.pop_back();
            boundary_listX.pop_back();
            boundary_listY.pop_back();
            break;
        }
        // detect cycle
        if ((lx1 == lx3)&&(ly1 == ly3)&&(lx2 == lx4)&&(ly2 == ly4)){
          boundary_listX.pop_back();
          boundary_listY.pop_back();
          boundary_listX.pop_back();
          boundary_listY.pop_back();
          // change direction from M_PI to 3*M_PI/2
          if ((DX == 0)&&(DY == 1)){
            DX = -1;
            DY = 0;
          }
          // from M_PI/2 to M_PI
          else if ((DX == 1)&&(DY == 0)){
            DX = 0;
            DY = 1;
          }
          // from 0 to M_PI/2
          else if ((DX == 0)&&(DY == -1)){
            DX = 1;
            DY = 0;
          }
          else{
            DX = 0;
            DY = -1;
          }
        }
      }
    }

    std::vector <std::vector<int> > boundary(2, std::vector<int>(sizeofX));

    boundary[0].assign(boundary_listX.begin(), boundary_listX.end());
    boundary[1].assign(boundary_listY.begin(), boundary_listY.end());

    return boundary;
}
