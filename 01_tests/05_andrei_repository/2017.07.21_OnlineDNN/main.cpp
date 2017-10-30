#include <unistd.h>

#include "utils.hpp"
#include "range.hpp"
#include "online_dnn_engine.hpp"

using namespace std;

int main() {

	OnlineDnnUtils *d = new OnlineDnnUtils();

	for (auto pair : zip(range(1,5), range(10, 50)))
		cout << get<0>(pair) << " " << get<1>(pair) << endl;

	std::vector<int> v1 = {1, 2, 3};
	std::vector<int> v2 = {10, 20, 30};

	auto a = zip(v1, v2);

	for (int i = 0 ; i < 100000; i++){
		sleep(0.01);
		print_progress((i+1)/100000.0);
	}

	printf("\n");

	MatrixXd m = MatrixXd(4,3);
	m << 1,7,10,-2,5,6,19,4,3,14,3,-4;
	cout << "m=\n" << m;

	NormData res = d->feature_normalize(m);
	cout << '\n' <<  res.X_norm << "\n\n" << res.min_val << endl;
	return 0;
}