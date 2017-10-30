#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

template<typename T>
std::vector<T> range(T min = 0, T max, int step = 1)
{
	std::vector<T> result;

	for (T i = min ; i < max ; i += step)
		result.push_back(i);

	return result;
}

int main(){

	auto thresholds = VectorXd::LinSpaced(10,1, 0);
	cout << thresholds << endl;

	MatrixXd v(2,3);

	v << 1,2,3,4,5,6;

	MatrixXd vec =  v.block(0, 0, 1, v.cols());

	cout << v << endl << endl;

	cout << vec << endl;
}