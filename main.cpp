#include <iostream>

using namespace std;

#define N 100

class Configuration {
public:
    Configuration generateNext(){
        return this;
    }

    double getMagnet(){
        return 0;
    }
};

int main()
{
    Configuration S;
    double sum = 0;
    for (int i = 0; i<N; i++){
        double M = S.getMagnet();
        sum += M;
        S = S.generateNext();
    }
    cout << sum / N;
}
