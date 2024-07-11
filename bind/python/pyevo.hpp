// mylib.h
#include <iostream>
#include <cmath>

class MyClass {
    double a;
    double b;
    double c;
public:

    MyClass() {}
    MyClass(double a_in, double b_in) {
        a = a_in;
        b = b_in;
        c = 0;
    }

    void run() {
        c = std::sqrt(a*a + b*b);
        std::cout<<c<<std::endl;
    }
};
