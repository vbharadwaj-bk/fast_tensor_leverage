#include <iostream>
#include "oneapi/mkl.hpp"

using namespace std;

int main(int argc, char** argv) {
    cout << "Hello world!" << endl;
    double* x = (double*) malloc(5 * sizeof(double)); 
    double* y = (double*) malloc(5 * sizeof(double)); 
    double res = cblas_ddot(5, x, 1, y, 1);
    cout << res << endl;
}