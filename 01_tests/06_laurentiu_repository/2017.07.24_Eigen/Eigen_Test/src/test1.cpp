#include <iostream>
#include <Eigen/Dense>
#include <vector>
using namespace Eigen;
using namespace std;

int main()
{
  vector<int> array;

  array.push_back(10);
  array.push_back(20);
  array.push_back(30);

  for (auto elem : array)
    cout << elem << endl;

  MatrixXd m = MatrixXd::Random(3,3);
  m = (m + MatrixXd::Constant(3,3,1.2)) * 50;
  cout << "m =" << endl << m << endl;
  VectorXd v(3);
  v << 1, 2, 3;
  cout << "m * v =" << endl << m * v << endl;
}