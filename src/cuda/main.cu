#include <iostream> // cout & endl
#include <getopt.h>

#include <cstdlib> // atoi(), atof()
#include <string> // stoi();
#include <fstream> // read and write from/to files
#include <sstream>
#include <iterator>


#include <ctime> // time()
#include <cfloat> // DBL_MAX
#include <limits> // FLT_MAX, DBL_MAX
#include <cmath> // pow()

#define TRAIN_SIZE 40000
#define TEST_SIZE 2000

typedef struct {
  uint8_t** train_set;
  uint8_t** test_set;
  uint8_t* train_labels;
  uint8_t* test_labels;
  int train_size;
  int test_size;
  int nFeatures;
  int nClasses;
} Dataset; 

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

__global__ void cuda_update_weights(int num_features, double* weight_vector, uint8_t* features, uint8_t label,
    double* probabilities, double* max, double* total) {
  int offset = threadIdx.x * num_features;      
  double probability = 0;

  for (int i = 0; i < num_features; i++) {
    probability += weight_vector[i + offset] * features[i];
  }
  probabilities[threadIdx.x] = probability;
  __syncthreads();

  if (threadIdx.x == 0) {
    *max = DBL_MIN;
    *total = 0;
    for (int i = 0; i < blockDim.x; i++) {
      if (probabilities[i] > *max) {
        *max = probabilities[i];
      }
    }
  }
  __syncthreads();

  probability = exp(probability - *max);
  atomicAdd(total, probability);

  __syncthreads();

  probability /= *total;

  double y = (threadIdx.x == label) ? 1 : 0;
  for (int i = 0; i < num_features; i++) {
    weight_vector[i + offset] += (y - probability) * features[i];
  }
}

double test(Dataset*, double**);

double** train(Dataset* ds) {
  // Generate random weight vector (784).
  double** weight_vectors = generate_k_weight_vectors(ds->nClasses, ds->nFeatures);
  printf("%f\n", test(ds, weight_vectors));

  // Malloc weight vectors and dataset arrays on GPU
  double* d_weight_vectors;
  cudaMalloc(&d_weight_vectors, ds->nClasses * ds->nFeatures * sizeof(double));
  for (int i = 0; i < ds->nClasses; i++) {
    int offset = i * ds->nFeatures;
    cudaMemcpy(d_weight_vectors + offset, weight_vectors[i], 
        ds->nFeatures * sizeof(double), cudaMemcpyHostToDevice);
  }
  uint8_t* d_train_set;
  cudaMalloc(&d_train_set, ds->train_size * ds->nFeatures * sizeof(uint8_t));
  for (int i = 0; i < ds->train_size; i++) {
    int offset = i * ds->nFeatures;
    cudaMemcpy(d_train_set + offset, ds->train_set[i],
        ds->nFeatures * sizeof(uint8_t), cudaMemcpyHostToDevice);
  }

  double* probabilities;
  cudaMalloc(&probabilities, ds->nClasses * sizeof(double));

  double* max;
  cudaMalloc(&max, sizeof(double));
  double* total;
  cudaMalloc(&total, sizeof(double));

  // For each training point:
  // 1. Calculate gradient.
  // 2. Update weight vector.
  // Continue through entire dataset.
  for (int i = 0; i < ds->train_size; i++) {
    cuda_update_weights<<<1, ds->nClasses>>>(ds->nFeatures, d_weight_vectors, &d_train_set[i * ds->nFeatures], 
      ds->train_labels[i], probabilities, max, total);
  }

  for (int i = 0; i < ds->nClasses; i++) {
    int offset = i * ds->nFeatures;
    cudaMemcpy(weight_vectors[i], d_weight_vectors + offset,
      ds->nFeatures * sizeof(double), cudaMemcpyDeviceToHost);
  }

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

  double** weight_vectors = train(ds);
  printf("%f\n", test(ds, weight_vectors));
}