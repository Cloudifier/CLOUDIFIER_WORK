#include <stdio.h>
#include <string>
#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <numeric>

#include <Eigen/Dense>

using namespace std;
using namespace std::chrono;
using namespace Eigen;

template <typename T>
std::ostream& operator<<( std::ostream& stream, const std::vector<T>& vect ) {
	for (unsigned int i = 0; i < vect.size(); i++) {
		stream << vect[i] << ' ';
	}
	stream << '\n';

	return stream;
}

struct TrainCrossSplits
{
	MatrixXd X_train;
	MatrixXd X_cross;
	vector <string> labels;
	VectorXd y_train;
	VectorXd y_cross;
};


class OnlineDnnBase {

private:
	long loaded_data_nr_fields;
	long loaded_data_nr_rows;
	long train_test_split_pos;

protected:
	milliseconds start_time;
	milliseconds end_time;
	default_random_engine random_engine;

	bool b_bias_added; // variable that stores bias information for pre-loaded data

	string CLF_NAME;
	long NR_FEATS;
	long NR_CLASSES;

public:
	bool VERBOSE_ENGINE;

	MatrixXd *X_loaded;
	VectorXd *y_loaded;

	MatrixXd *X_train;
	VectorXd *y_train;

	MatrixXd *X_cross;
	VectorXd *y_cross;

	MatrixXd *loaded_data;
	vector <string> loaded_data_header;
	vector <string> labels_vector;
	vector <int> label_ids_vector;

	bool file_exists(const string& name);
	MatrixXd shuffle_matrix_rows(MatrixXd data_matrix);
	TrainCrossSplits load_csv(const string & inputfile, const bool b_shuffle,
		const bool b_add_bias);

	void begin_timer();
	long end_timer();

	int find_label_id(vector <string> labels, string value);
	vector <string> to_labels(VectorXd y);

	OnlineDnnBase()
	{
		CLF_NAME = "Online DNN Base";

		unsigned seed = system_clock::now().time_since_epoch().count();
		random_engine = default_random_engine(seed);
		VERBOSE_ENGINE = false;
		b_bias_added = false;
		X_train = NULL;
		y_train = NULL;
		X_loaded = NULL;
		y_loaded = NULL;
		X_cross = NULL;
		y_cross = NULL;
		loaded_data = NULL;
	}

	virtual ~OnlineDnnBase()
	{
		debug_info("OnlineDnnBase destructor called");

		if (loaded_data != NULL)
			delete loaded_data;
		if (X_loaded != NULL)
			delete X_loaded;
		if (y_loaded != NULL)
			delete y_loaded;
		if (X_train != NULL)
			delete X_train;
		if (y_train != NULL)
			delete y_train;
		if (X_cross != NULL)
			delete X_cross;
		if (y_cross != NULL)
			delete y_cross;
	}

	void debug_info(string str_message)
	{
		if (VERBOSE_ENGINE)
			printf("\n[DEBUG] %s", str_message.c_str());
	}

	void debug_info(string msg, MatrixXd mat)
	{
		std::stringstream ss;
		ss << mat;
		string str_matrix = ss.str();
		string msgp = "\n[DEBUG] " + msg + "\n";
		cout << msgp ;
		cout << str_matrix << endl;
	}

	void debug_info(MatrixXd mat)
	{
		std::stringstream ss;
		ss << mat;
		string str_matrix = ss.str();
		printf("\n[DEBUG] Matrix:\n");
		cout << str_matrix << endl;
	}

	void debug_info(VectorXd vec, bool b_horizontal)
	{
		std::stringstream ss;
		if (b_horizontal)
			ss << vec.transpose();
		else
			ss << vec;
		string str_vector = ss.str();
		printf("\n[DEBUG] Vector:\n");
		cout << str_vector << endl;
	}

	void debug_info(string msg, VectorXd vec, bool b_horizontal)
	{
		std::stringstream ss;
		if (b_horizontal)
			ss << vec.transpose();
		else
			ss << vec;
		string str_vector = ss.str();
		string msgp = "\n[DEBUG] " + msg + "\n";
		cout << msgp;
		cout << str_vector << endl;
	}
};


inline bool OnlineDnnBase::file_exists(const string& name)
{
	if (FILE *file = fopen(name.c_str(), "r")) {
		fclose(file);
		return true;
	}
	else {
		return false;
	}
}

void OnlineDnnBase::begin_timer()
{
	milliseconds ms = duration_cast< milliseconds > (system_clock::now().time_since_epoch());
	start_time = ms;
}

inline long OnlineDnnBase::end_timer()
{
	milliseconds ms = duration_cast< milliseconds > (system_clock::now().time_since_epoch());
	end_time = ms;
	return (end_time - start_time).count();
}

inline int OnlineDnnBase::find_label_id(vector<string> labels, string value)
{
	uint pos = find(labels.begin(), labels.end(), value) - labels.begin();

	if (pos >= labels.size())
		return -1;

	return pos;
}

inline vector<string> OnlineDnnBase::to_labels(VectorXd y)
{
	vector <string> labels;
	for (long i = 0; i < y.size(); i++) {
		string s = labels_vector[y(i)];
		labels.push_back(s);
	}
	return labels;
}

inline TrainCrossSplits OnlineDnnBase::load_csv(const string & inputfile,
	const bool b_shuffle, const bool b_add_bias)
{

	int nr_rows = 0;
	int nr_cols = 0;
	string fname = inputfile;
	TrainCrossSplits rec_results;

	if (!file_exists(inputfile))
		throw std::invalid_argument("Received invalid file in load_csv: " + fname);

	ifstream infile(fname, std::ifstream::in);

	if (!infile.good())
		throw std::invalid_argument("Received invalid file in load_csv: " + fname);

	debug_info("Loading " + fname + " dataset...");
	vector< vector<string> > result;
	while (!infile.eof()) {
		//go through every line
		string line;

		getline(infile, line);

		vector <string> record;
		nr_cols = 0;

		std::size_t prev = 0, pos;
		while ((pos = line.find_first_of(",;", prev)) != std::string::npos) {
			if (pos > prev) {
				record.push_back(line.substr(prev, pos - prev));
				nr_cols++;
			}

			prev = pos + 1;
		}
		if (prev < line.length()) {
			record.push_back(line.substr(prev, std::string::npos));
			nr_cols++;
		}


		if (nr_cols > 0) {
			result.push_back(record);
			nr_rows++;
		}
	}


	//
	// now load whole data, X and y matrices
	// assume last column of loaded data is the results / labels
	//
	loaded_data_nr_fields = result[0].size();
	loaded_data_nr_rows = nr_rows - 1; // rows minus field names row

	debug_info("Loaded " + std::to_string(loaded_data_nr_rows) + " X " + std::to_string(loaded_data_nr_fields) + " dataset");

	loaded_data = new MatrixXd(loaded_data_nr_rows, loaded_data_nr_fields);
	y_loaded = new VectorXd(loaded_data_nr_rows);
	X_loaded = new MatrixXd(loaded_data_nr_rows, loaded_data_nr_fields - 1);

	std::set <string> labels_set;

	long i, j;
	for (j = 0; j < loaded_data_nr_fields; j++)
		loaded_data_header.push_back((string)result[0][j]);

	//
	// assume dataset is curated and ONLY last column contains text labels
	//
	vector <string> loaded_labels;

	for (i = 0; i < loaded_data_nr_rows; i++) {
		for (j = 0; j < loaded_data_nr_fields; j++) {
			double fcell = 0;
			string scell = result[i + 1][j];
			try {
				if (j != ((loaded_data_nr_fields - 1)))
					fcell = ::atof(scell.c_str());
			}
			catch (...) {

			}
			(*loaded_data)(i, j) = fcell;
			if (j == (loaded_data_nr_fields - 1)) {
				labels_set.insert(scell);
				loaded_labels.push_back(scell);
			}
		}
	}

	labels_vector.assign(labels_set.begin(), labels_set.end());

	std::set <int> label_ids_set;
	for (uint label_idx = 0; label_idx < loaded_labels.size(); label_idx++) {
		string c_label = loaded_labels[label_idx];
		int i_label = find_label_id(labels_vector, c_label);
		label_ids_set.insert(i_label);
		(*loaded_data)(label_idx, loaded_data_nr_fields - 1) = i_label;
	}

	label_ids_vector.assign(label_ids_set.begin(), label_ids_set.end());

	if (b_shuffle) {
		MatrixXd ttt = shuffle_matrix_rows(*loaded_data);
		*loaded_data = ttt;
	}

	float test_size = 0.2;
	int test_rows = loaded_data_nr_rows * test_size;
	int train_rows = loaded_data_nr_rows - test_rows;
	train_test_split_pos = train_rows;

	*X_loaded = loaded_data->leftCols(loaded_data_nr_fields - 1);
	*y_loaded = loaded_data->rightCols(1);

	NR_FEATS = X_loaded->cols();
	NR_CLASSES = labels_vector.size();

	if (b_add_bias) {
		// now add bias
		VectorXd bias(loaded_data_nr_rows);
		bias.fill(1);
		MatrixXd *TempX = new MatrixXd(loaded_data_nr_rows, loaded_data_nr_fields);
		*TempX << bias, *X_loaded;
		b_bias_added = true;
		delete X_loaded;
		X_loaded = TempX;
		// done adding bias
	}

	X_train = new MatrixXd(X_loaded->topRows(train_rows));
	X_cross = new MatrixXd(X_loaded->bottomRows(test_rows));

	y_train = new VectorXd(y_loaded->head(train_rows));
	y_cross = new VectorXd(y_loaded->tail(test_rows));

	rec_results.X_cross = *X_cross;
	rec_results.X_train = *X_train;
	rec_results.y_cross = *y_cross;
	rec_results.y_train = *y_train;
	rec_results.labels = labels_vector;

	return(rec_results);
}

inline MatrixXd OnlineDnnBase::shuffle_matrix_rows(MatrixXd data_matrix)
{
	long size = data_matrix.rows();
	PermutationMatrix<Dynamic, Dynamic> perm(size);
	perm.setIdentity();

	std::shuffle(perm.indices().data(),
				 perm.indices().data() + perm.indices().size(),
				 this->random_engine);

	MatrixXd A_perm = perm * data_matrix; // permute rows
	return(A_perm);
}


struct NormData {
	MatrixXd X_norm;
	RowVectorXd min_val;
	RowVectorXd div_val;
};

class OnlineDnnUtils : public OnlineDnnBase {
public:
	OnlineDnnUtils()
	{
		CLF_NAME = "Online DNN Utils";
	}

	~OnlineDnnUtils()
	{
		debug_info("Deleting object [" + CLF_NAME + "]");
	}

	NormData feature_normalize(MatrixXd X_data, string method = string("z-score"));
	MatrixXd test_data_normalize(MatrixXd X_data, RowVectorXd min_val, RowVectorXd div_val);
	double kappa(VectorXd y_pred, VectorXd y_truth);

private:
	struct BinaryAnd {
 		
 		EIGEN_EMPTY_STRUCT_CTOR(BinaryAnd)
 		bool operator()(const bool& a, const bool& b) const { return a&b; }
	};
};

NormData OnlineDnnUtils::feature_normalize(MatrixXd X_data, string method)
{
	NormData rec_results;
	RowVectorXd min_val;
	RowVectorXd div_val;
	if (method == "z-score") {
		min_val = X_data.colwise().mean();
		div_val = ((X_data.rowwise() - min_val).array().square().colwise().sum() /
			X_data.rows()).sqrt();
	} else if (method == "minmax") {
		min_val = X_data.colwise().minCoeff();
		div_val = X_data.colwise().maxCoeff() - min_val;
	} else
		throw std::invalid_argument("[feature_normalize] Unknown scale/norm method: " + method);

	div_val = div_val.unaryExpr([](double x) { return x == 0.0 ? 1.0 : x; });

	MatrixXd result = (X_data.rowwise() - min_val).array().rowwise() / div_val.array();
	rec_results.X_norm = result;
	rec_results.min_val = min_val;
	rec_results.div_val = div_val;

	return rec_results;
}

MatrixXd OnlineDnnUtils::test_data_normalize(MatrixXd X_test, RowVectorXd min_val,
	RowVectorXd div_val)
{
	return (X_test.rowwise() - min_val).array().rowwise() / div_val.array();
}

double OnlineDnnUtils::kappa(VectorXd y_pred, VectorXd y_truth)
{
	NR_CLASSES = 2;

	vector <unsigned int> TP(NR_CLASSES, 0);
	vector <unsigned int> FP(NR_CLASSES, 0);
	vector <unsigned int> TN(NR_CLASSES, 0);
	vector <unsigned int> FN(NR_CLASSES, 0);
	VectorXi class_pred = VectorXi::Zero(NR_CLASSES);
	VectorXi class_real = VectorXi::Zero(NR_CLASSES);


	Matrix<bool, Dynamic, 1> m1, m2;
	for (unsigned int i = 0; i < NR_CLASSES; ++i) {
		{
			m1 = y_pred.array() == (double) i;
			m2 = y_truth.array() == (double) i;
			TP[i] = m1.binaryExpr(m2, BinaryAnd()).count();
		}

		{
			m1 = y_pred.array() != (double) i;
			m2 = y_truth.array() != (double) i;
			TN[i] = m1.binaryExpr(m2, BinaryAnd()).count();
		}

		{
			m1 = y_pred.array() == (double) i;
			m2 = y_truth.array() != (double) i;
			FP[i] = m1.binaryExpr(m2, BinaryAnd()).count();
		}
	
		{
			m1 = y_pred.array() != (double) i;
			m2 = y_truth.array() == (double) i;
			FN[i] = m1.binaryExpr(m2, BinaryAnd()).count();
		}

		class_pred(i) = TP[i] + FP[i];
        class_real(i) = TP[i] + FN[i];
	}

	unsigned int all_ex = TP[0] + TN[0] + FP[0] + FN[0];
	cout << all_ex << endl;
	double obs_accuracy = ((double) std::accumulate(TP.begin(), TP.end(), 0)) / all_ex;
	cout << obs_accuracy << endl;
	
	// Check if expected_accuracy is good!!!!!!!
	double exp_accuracy = (((double) class_pred.cwiseProduct(class_real).sum()) / all_ex) / all_ex;
	cout << exp_accuracy<<endl;
 	double kappa = (obs_accuracy - exp_accuracy) / (1 - exp_accuracy);
	
	return kappa;
}


class OnlineDnnLayer : public OnlineDnnBase {
private:
	vector <string> valid_activations{ "", "direct", "sigmoid", "relu", "tanh", "softmax" };
	vector <string> valid_cost{ "cross_entropy", "MSE" };
	vector <string> valid_layers{ "", "input", "hidden", "output" };
	void init();

	struct ComputingOK {
 		EIGEN_EMPTY_STRUCT_CTOR(ComputingOK)
 		bool operator()(const double& a) const { return (std::isinf(a) || std::isnan(a)); }
	};

protected:
	string layer_name;
	int layer_id;
	string layer_type;
	int nr_units;
	string activation;
	OnlineDnnLayer *previous_layer;
	OnlineDnnLayer *next_layer;

	MatrixXd *theta;
	MatrixXd *theta_saved;
    MatrixXd *best_theta;

    string cost_function;
    bool is_output;
    int step;

public:
	OnlineDnnLayer()
	{
		CLF_NAME = "OnlineDnnLayer";
		init();
	}

	OnlineDnnLayer(int nr_units, string layer_name = string(""),
		OnlineDnnLayer *previous_layer = NULL, OnlineDnnLayer *next_layer = NULL,
		string activation = string(""), string layer_type = string(""),
		bool output = false) : OnlineDnnLayer()
	{
		debug_info("Deleting object [" + CLF_NAME + "]");
		this->layer_name = layer_name;
		this->nr_units = nr_units;
		this->layer_type = layer_type;
		this->activation = activation;

		/*if (std::find_if(begin(valid_layers), end(valid_layers),
     		[](const std::string& s){ return s == layer_type; }) != std::end(valid_layers))
     	{
     		throw std::invalid_argument("[OnlineDNNLayer: ERROR] unknown layer type: " +
     			layer_type);
     	}

     	if (std::find_if(begin(valid_activations), end(valid_activations),
     		[](const std::string& s){ return s == activation; }) != std::end(valid_activations))
     	{
     		throw std::invalid_argument("[OnlineDNNLayer: ERROR] unknown activation: " +
     			activation);
     	}*/

     	this->previous_layer = previous_layer;
     	this->next_layer = next_layer;
     	this->is_output = output;

	}

	~OnlineDnnLayer()
	{
		if (theta != NULL)
			delete theta;
		if (theta_saved != NULL)
			delete theta_saved;
		if (best_theta != NULL)
			delete best_theta;
		if (previous_layer != NULL)
			delete previous_layer;
		if (next_layer != NULL)
			delete next_layer;
	}

	bool is_computing_ok(MatrixXd z);
};

void OnlineDnnLayer::init()
{
	layer_id = -1;
	theta = NULL;
	theta_saved = NULL;
	best_theta = NULL;

	cost_function = "";
	step = -1;
}

inline bool OnlineDnnLayer::is_computing_ok(MatrixXd z)
{
	return !(z.unaryExpr(ComputingOK()).count() > 0);
}

