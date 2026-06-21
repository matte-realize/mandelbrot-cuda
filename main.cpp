#include <iostream>

int mandelbrotIterations(double cx, double cy, int maxIter) {
    double zx = 0.0, zy = 0.0;
    int iter = 0;
    
    while (zx*zx + zy*zy <= 4.0 && iter < maxIter) {
        double newZx = zx * zx - zy * zy + cx;
        double newZy = 2 * zx * zy + cy;
        zx = newZx;
        zy = newZy;
        iter++;
    }

    return iter;
}