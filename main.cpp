#include <iostream>
#include <fstream>

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

void iterToColor(
    int iter, 
    int maxIter, 
    unsigned char& r, 
    unsigned char& g,
    unsigned char& b
) {
    if (iter == maxIter) {
        r = g = b = 0;
        return;
    }

    double t = double (iter) / maxIter;

    r = (unsigned char) (9 * (1 - t) * t * t * t * 255);
    g = (unsigned char) (15 * (1 - t) * (1 - t) * t * t * 255);
    b = (unsigned char) (8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
}

int main() {
    int width = 800, height = 600;
    int maxIter = 256;
    double xmin = -0.75, xmax = -0.74;
    double ymin = 0.1, ymax = 0.11;

    std::ofstream out("output/frame.ppm", std::ios::binary);
    out << "P6\n" << width << " " << height << "\n255\n";

    for (int py = 0; py < height; py++) {
        for (int px = 0; px < width; px++) {
            double cr = xmin + (px / (double) width) * (xmax - xmin);
            double ci = ymin + (py / (double) height) * (ymax - ymin);
            
            int iter = mandelbrotIterations(cr, ci, maxIter);
            
            unsigned char r, g, b;

            iterToColor(iter, maxIter, r, g, b);

            out.write((char*) &r, 1);
            out.write((char*) &g, 1);
            out.write((char*) &b, 1);
        }
    }
}