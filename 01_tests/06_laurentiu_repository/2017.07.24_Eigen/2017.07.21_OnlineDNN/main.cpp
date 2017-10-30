#include "online_dnn_engine.hpp"



int main() {
//	OnlineDnnUtils *d = new OnlineDnnUtils();
	
	/*MatrixXd m = MatrixXd(2,2);

	m << 1,2,3,4;
	MatrixXd n = MatrixXd(m);
	n(0) = 7000;

	cout << m << endl;
	cout << n << endl;*/

	vector <string> valid_activations{ "", "direct", "sigmoid", "relu", "tanh", "softmax" };
	vector <string> valid_cost{ "cross_entropy", "MSE" };
	vector <string> valid_layers{ "", "input", "hidden", "output" };

	string layer_type = "direct";

	if (std::find(valid_activations.begin(), valid_activations.end(), layer_type) != valid_activations.end()) {
		cout << "DA";
	}


	return 0;
}