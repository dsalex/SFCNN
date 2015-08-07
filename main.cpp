#include <iostream>
#include <armadillo>

using namespace std;

int main()
{
    arma::mat A = arma::randu<arma::mat>(4,5);
    arma::mat B = arma::randu<arma::mat>(4,5);

    cout << A*B.t() << endl;
    cout << "Hello World!" << endl;
    return 0;
}

