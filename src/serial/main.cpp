#include <iostream> // cout & endl
#include <getopt.h>

#include <cstdlib> // atoi(), atof()
#include <string> // stoi();
#include <fstream> // read and write from/to files
#include <sstream>
#include <iterator>


#include <ctime> // time()
#include <chrono>
#include <cfloat> // DBL_MAX
#include <limits> // FLT_MAX, DBL_MAX
#include <cmath> // pow()

#define TRAIN_SIZE 40000
#define TEST_SIZE 2000

struct Dataset {
  uint8_t** train_set;
  uint8_t** test_set;
  uint8_t* train_labels;
  uint8_t* test_labels;
  int train_size;
  int test_size;
  int nFeatures;
  int nClasses;

  ~Dataset() {

  }
};

Dataset* preprocess_data(Dataset*);

Dataset* load_data() {
  // Initialize data set
  Dataset* ds = new Dataset();
  ds->train_size = TRAIN_SIZE;
  ds->test_size = TEST_SIZE;
  ds->nFeatures = 784;
  ds->nClasses = 10;
  ds->train_set = new uint8_t*[ds->train_size];
  for (int i = 0; i < ds->train_size; i++) {
    ds->train_set[i] = new uint8_t[ds->nFeatures];
  }
  ds->test_set = new uint8_t*[ds->test_size];
  for (int i = 0; i < ds->test_size; i++) {
    ds->test_set[i] = new uint8_t[ds->nFeatures];
  }
  ds->train_labels = new uint8_t[ds->train_size];
  ds->test_labels = new uint8_t[ds->test_size];
  
  // 42001 rows × 785 columns
  // First row represents column headers
  // First column represents image labels
  std::ifstream input_file;
  input_file.open("train.csv");

  std::string line;
  getline(input_file, line); // Remove first line with column headers

  // Read in train data
  for (int i = 0; i < ds->train_size; i++) {
    getline(input_file, line, ',');
    ds->train_labels[i] = stoi(line);

    for (int j = 0; j < ds->nFeatures - 1; j++) {
      getline(input_file, line, ',');
      ds->train_set[i][j] = stoi(line);
    }
    
    getline(input_file, line);
    ds->train_set[i][ds->nFeatures - 1] = stoi(line);
  }

  // Read in test data
  for (int i = 0; i < ds->test_size; i++) {
    getline(input_file, line, ',');
    ds->test_labels[i] = stoi(line);

    for (int j = 0; j < ds->nFeatures - 1; j++) {
      getline(input_file, line, ',');
      ds->test_set[i][j] = stoi(line);
    }
    
    getline(input_file, line);
    ds->test_set[i][ds->nFeatures - 1] = stoi(line);
  }  

  input_file.close();

  //ds = preprocess_data(ds);

  return ds;
}

int convert_2d_to_1d(int x, int y) {
  return x * 28 + y;
}


int xdir[4] = {0, -1, 0, 1};
int ydir[4] = {-1, 0, 1, 0};

int find_connected_component(bool** arr, int x, int y) {
  int xdir[4] = {0, -1, 0, 1};
  int ydir[4] = {-1, 0, 1, 0};

  arr[x][y] = 1;

  int size = 1;

  for (int i = 0; i < 4; i++) {
    int x2 = x + xdir[i];
    int y2 = y + ydir[i];
    if (x2 >= 0 && x2 < 28 && y2 >= 0 && y2 < 28 && !arr[x2][y2]) {
      size += find_connected_component(arr, x2, y2);
    }
  }

  return size;
}

int count_connected_components(uint8_t* old_features) {
  //bool** arr = new bool[28][28];
  bool** arr = new bool*[28];
  for (int i = 0; i < 28; i++) {
    arr[i] = new bool[28];
  }
  int connected_components = 0;

  // Convert grayscale image to black and white boolean matrix
  for (int x = 0; x < 28; x++) {
    for (int y = 0; y < 28; y++) {
      arr[x][y] = old_features[convert_2d_to_1d(x, y)] > 128; // False if white
    }
  }

  for (int x = 0; x < 28; x++) {
    for (int y = 0; y < 28; y++) {
      if (!arr[x][y]) {
        int size = find_connected_component(arr, x, y);
        if (size >= 10) {
          connected_components++;
        }
      }
    }
  }

  for (int i = 0; i < 28; i++) {
    delete[] arr[i];
  }
  delete[] arr;

  //printf("%d connected_components\n", connected_components);
  return connected_components;
}


void apply_stencil(uint8_t* new_features, uint8_t* old_features) {
  for (int x = 0; x < 28; x++) {
    for (int y = 0; y < 28; y++) {
      uint8_t average = 0;
      int count = 0;
      for (int i = 0; i < 4; i++) {
        int x2 = x + xdir[i];
        int y2 = y + ydir[i];
        if (x2 >= 0 && x2 < 28 && y2 >= 0 && y2 < 28) {
          int index = convert_2d_to_1d(x2, y2);
          average += old_features[index];
          count++;
        }
      }
      average /= count;
      new_features[convert_2d_to_1d(x, y)] = average;
    }
  }
}

void preprocess_data2(Dataset* data) {
  Dataset* ds = new Dataset();
  ds->train_size = data->train_size;
  ds->test_size = data->test_size;
  ds->nFeatures = data->nFeatures + 1;
  ds->nClasses = data->nClasses;

  ds->train_labels = data->train_labels;
  ds->test_labels = data->test_labels;


  auto start = std::chrono::high_resolution_clock::now();
  
  ds->train_set = new uint8_t*[ds->train_size];
  for(int i = 0; i < ds->train_size; i++) {
    ds->train_set[i] = new uint8_t[ds->nFeatures];
    apply_stencil(ds->train_set[i], data->train_set[i]);
    //printf("%d ", ds->train_set[i][data->nFeatures * scale]);
    //delete data->train_set[i];
  }

  // for (int i = 0; i < ds->train_size; i++) {
  //   ds->train_set[i][ds->nFeatures - 1] = count_connected_components(data->train_set[i]);
  // }
  //delete data->train_set;


  ds->test_set = new uint8_t*[ds->test_size];
  for(int i = 0; i < ds->test_size; i++) {
    ds->test_set[i] = new uint8_t[ds->nFeatures];
    apply_stencil(ds->test_set[i], data->test_set[i]);
    //delete data->test_set[i];
  }
  //delete data->test_set;

  // for (int i = 0; i < ds->test_size; i++) {
  //   ds->test_set[i][ds->nFeatures -1] = count_connected_components(data->test_set[i]);
  // }


  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  long long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  printf("%llu ms\n", milliseconds);
}

void expand_features(int factor, int new_feature_size, int old_feature_size, uint8_t* new_features, uint8_t* old_features) {
  for (int i = 0; i < old_feature_size; i++) {
    for (int j = 0; j < factor; j++) {
      new_features[i * factor + j] = old_features[i] > 0;
    }
  }
}

Dataset* preprocess_data(Dataset* data) {
  int scale = 1;

  Dataset* ds = new Dataset();
  ds->train_size = data->train_size;
  ds->test_size = data->test_size;
  ds->nFeatures = data->nFeatures * scale + 1;
  ds->nClasses = data->nClasses;

  ds->train_set = new uint8_t*[ds->train_size];
  for(int i = 0; i < ds->train_size; i++) {
    ds->train_set[i] = new uint8_t[ds->nFeatures];
    expand_features(scale, ds->nFeatures, data->nFeatures, ds->train_set[i], data->train_set[i]);
    ds->train_set[i][data->nFeatures * scale] = count_connected_components(data->train_set[i]);
    //printf("%d ", ds->train_set[i][data->nFeatures * scale]);
    delete data->train_set[i];
  }
  delete data->train_set;

  ds->train_labels = data->train_labels;

  ds->test_set = new uint8_t*[ds->test_size];
  for(int i = 0; i < ds->test_size; i++) {
    ds->test_set[i] = new uint8_t[ds->nFeatures];
    expand_features(scale, ds->nFeatures, data->nFeatures, ds->test_set[i], data->test_set[i]);
    ds->test_set[i][data->nFeatures * scale] = count_connected_components(data->test_set[i]);
    delete data->test_set[i];
  }
  delete data->test_set;

  ds->test_labels = data->test_labels;

  //delete data;

  return ds;
}

double* generate_weight_vector(int size) {
  double* w = new double[size];

  for(int i = 0; i < size; i++) {
    w[i] = (double) std::rand() / RAND_MAX;
  }
  return w;
}

double** generate_k_weight_vectors(int k, int size) {
  double** w = new double*[k];
  for (int i = 0; i < k; i++) {
    w[i] = generate_weight_vector(size);
  }
  return w;
}

double inner_product(int len, double* w, uint8_t* f) {
  double sum = 0;
  for (int i = 0; i < len; i++) {
    sum += w[i] * f[i];
  }

  return sum;
}

uint8_t predict(int num_classes, int num_features, double** weight_vectors, uint8_t* features) {
  double max = 0;
  uint8_t label = 0;
  for (int i = 0; i < num_classes; i++) {
    double product = inner_product(num_features, weight_vectors[i], features);
    if (product > max) {
      max = product;
      label = i;
    }
  }
  return label; 
}

void update_weights(Dataset* ds, double** weight_vectors, uint8_t* features, uint8_t label) {
  double* probabilities = new double[ds->nClasses];
  double max_product = DBL_MIN;

  for (int i = 0; i < ds->nClasses; i++) {
    double product = inner_product(ds->nFeatures, weight_vectors[i], features);
    probabilities[i] = product;
    if (product > max_product) {
      max_product = product;
    }
  } 

  // Exponentiate and calculate sum.
  double sum = 0;

  for (int i = 0; i < ds->nClasses; i++) {
    probabilities[i] = std::exp(probabilities[i] - max_product);
    sum += probabilities[i];
  }

  // Calculate probabilities.
  for (int i = 0; i < ds->nClasses; i++) {
    probabilities[i] /= sum;
  }

  // Update weight vectors.
  for (int i = 0; i < ds->nClasses; i++) {
    double y = (i == label) ? 1 : 0;
    for (int j = 0; j < ds->nFeatures; j++) {
      weight_vectors[i][j] += (y - probabilities[i]) * features[j];
    }
  }
}
double test(Dataset*, double**);

double** train(Dataset* ds) {
  // Generate random weight vector (784).
  double** weight_vectors = generate_k_weight_vectors(ds->nClasses, ds->nFeatures);
  printf("%f\n", test(ds, weight_vectors));

  auto start = std::chrono::high_resolution_clock::now();

  // For each training point:
  // 1. Calculate gradient.
  // 2. Update weight vector.
  // Continue through entire dataset.
  for (int i = 0; i < ds->train_size; i++) {
    //uint8_t prediction = predict(10, 784, weight_vectors, train_set[i]);
    //int index = i % ds->train_size;
    update_weights(ds, weight_vectors, ds->train_set[i], ds->train_labels[i]);
  }

  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  long long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  printf("%llu ms\n", milliseconds);

  return weight_vectors;
}

double test(Dataset* ds, double** weight_vectors) {
  // For each training point, generate prediction.
  int correct = 0;
  for (int i = 0; i < ds->test_size; i++) {
    uint8_t prediction = predict(ds->nClasses, ds->nFeatures, weight_vectors, ds->test_set[i]);
    correct += (prediction == ds->test_labels[i]);
  }

  // Compute total accuracy.
  double accuracy = (double) correct / ds->test_size;
  return accuracy;
}


int main(int argc, char *argv[]) {
  std::srand(std::time(nullptr));

  Dataset* ds = load_data();
  preprocess_data2(ds);

  // ds = preprocess_data(ds);

  // //printf("%d\n", ds->nFeatures);

  // double** weight_vectors = train(ds);
  // printf("%f\n", test(ds, weight_vectors));
}
