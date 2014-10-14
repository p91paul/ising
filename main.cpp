#include <iostream>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <ctime>
#include <cmath>

using namespace std;
using namespace boost::random;

#define N 10000
#define L 32
#define L3 L*L*L
#define J 1
#define B 0
#define T 16.0

class Configuration {
public:
    Configuration(int seed = time(0)){
        gen.seed(seed);
        for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
        for (int k = 0; k < L; k++){
            S[i][j][k] = randomSpin();
        }
    }

    void generateNext(){
        const double beta = 1/T;
        for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
        for (int k = 0; k < L; k++){
            int dE = -2 * S[i][j][k] * neigh(i,j,k);
            if (dE < 0 || pdist(gen) < exp(-beta*dE))
                S[i][j][k] *= -1;
        }
    }

    double getMagnet(){
        double sum = 0;
        for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
        for (int k = 0; k < L; k++){
            sum += S[i][j][k];
        }
        return sum;
    }
private:
    int S[L][L][L];
    mt19937 gen;
    uniform_int_distribution<> spindist = uniform_int_distribution<>(0, 1);
    uniform_real_distribution<> pdist = uniform_real_distribution<>(0, 1);

    inline int randomSpin(){
        return 2*spindist(gen)-1;
    }

    inline int neigh(int i, int j, int k){
        return S[i][j][(k+1)%L] +
               S[i][j][(k+L-1)%L] +
               S[i][(j+1)%L][k] +
               S[i][(j+L-1)%L][k] +
               S[(i+1)%L][j][k] +
               S[(i+L-1)%L][j][k];
    }
};

int main()
{
    Configuration S;
    double sum = 0;
    for (int i = 0; i<N; i++){
        S.generateNext();
        double M = S.getMagnet();
        sum += M;
    }
    cout << sum / N << endl;
}
