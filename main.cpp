/*
Sequential implementation of metropolis algorithm for Ising model
Copyright (C) 2014  Paolo Inaudi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <ctime>
#include <cmath>

using namespace std;
using namespace boost::random;

#define L 24
#define L3 L*L*L
#define J 1
#define B 0

class Configuration {
public:
    Configuration(double T, int seed = time(0)):T(T){
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
            if (dE <= 0 || pdist(gen) < exp(-beta*dE))
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
    const double T;

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

int main(int argc, char** argv)
{
    double T = 0;
    int N = 2;
    if (argc >= 2)
        T = atoi(argv[1]);
    if (argc >= 3)
        N = atoi(argv[2]);
    Configuration S(T, time(NULL));
    double sum = 0;
    for (int i = 0; i<N; i++){
        S.generateNext();
        double M = S.getMagnet();
        sum += M;
    }
    cout << sum / N << endl;
}
