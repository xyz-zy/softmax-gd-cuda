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
  ds = preprocess_data(ds);
  return ds;
}

void expand_features(int factor, int new_feature_size, int old_feature_size, uint8_t* new_features, uint8_t* old_features) {
  for (int i = 0; i < old_feature_size; i++) {
    for (int j = 0; j < factor; j++) {
      new_features[i * factor + j] = old_features[i];
    }
  }
}

int convert_2d_to_1d(int x, int y) {
  return x * 28 + y;
}

void find_connected_component(bool** arr, int x, int y) {
  int xdir[4] = {0, -1, 0, 1};
  int ydir[4] = {-1, 0, 1, 0};

  arr[x][y] = 1;

  for (int i = 0; i < 4; i++) {
    int x2 = x + xdir[i];
    int y2 = y + ydir[i];
    if (x2 >= 0 && x2 < 28 && y2 >= 0 && y2 < 28 && !arr[x2][y2]) {
      find_connected_component(arr, x2, y2);
    }
  }
}

int count_connected_components(uint8_t* old_features) {
  bool** arr = new bool*[28];
  for (int i = 0; i < 28; i++) {
    arr[i] = new bool[28];
  }
  int connected_components = 0;

  // Convert grayscale image to black and white boolean matrix
  for (int x = 0; x < 28; x++) {
    for (int y = 0; y < 28; y++) {
      arr[x][y] = old_features[convert_2d_to_1d(x, y)] >= 64;
    }
  }
  for (int x = 0; x < 28; x++) {
    for (int y = 0; y < 28; y++) {
      if (!arr[x][y]) {
        connected_components++;
      }
    }
  }

  for (int i = 0; i < 28; i++) {
    delete[] arr[i];
  }
  delete[] arr;

  return connected_components;
}

Dataset* preprocess_data(Dataset* data) {
  int scale = 2;

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

  delete data;

  return ds;
}
/*
Dataset* preprocess_data(Dataset* ds) {
  int num_features = 2 * ds->nFeatures + 1;

  for (int i = 0; i < ds->train_size; i++) {
    uint8_t* temp = new uint8_t[num_features];
    double_features(ds->nFeatures * 2, ds->nFeatures, temp, ds->train_set[i]);
    delete ds->train_set[i];
    ds->train_set[i] = temp;

    //ds->train_set[i][ds->nFeatures * 2] = count_connected_components(ds->train_set[i]);
  }

  for (int i = 0; i < ds->test_size; i++) {
    uint8_t* temp = new uint8_t[num_features];
    double_features(ds->nFeatures * 2, ds->nFeatures, temp, ds->test_set[i]);
    delete ds->test_set[i];
    ds->test_set[i] = temp;

    //ds->test_set[i][ds->nFeatures * 2] = count_connected_components(ds->test_set[i]);

  }

  ds->nFeatures = num_features;
  return ds;
}*/

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

// Call with <<<num_classes, power of 2 less than num_features>>>
__global__ void cuda_compute_probabilities(double* probabilities, double* weight_vectors, uint8_t* features, int num_features) {  
  extern __shared__ double temp[];
  int offset = num_features * blockIdx.x;
  int tid = threadIdx.x;
  int val = weight_vectors[offset + tid] * features[tid];
  if (blockDim.x + tid < num_features) {
    val += weight_vectors[offset + tid + blockDim.x] * features[tid + blockDim.x];
  }
  temp[tid] = val;
  __syncthreads();

  for (int step = blockDim.x/2; step > 0; step >>= 1) {
    if (tid < step) {
      temp[tid] += temp[tid + step];
      temp[tid + step] = -1;
    } 
    __syncthreads();
  }

  if (tid == 0) {
    probabilities[blockIdx.x] = temp[0];
  }
}

// Call with <<<1,1>>>
__global__ void cuda_find_max(int len, double* array, double* max, double* total) {
  double tmp_max = DBL_MIN;
  double tmp_total = 0;
  for (int i = 0; i < len; i++) {
    if (array[i] > tmp_max) {
      tmp_max = array[i];
    }
  }
  *max = tmp_max;

  for (int i = 0; i < len; i++) {
    array[i] = exp(array[i] - tmp_max);
    tmp_total += array[i];
  }

  *total = tmp_total;
}

__global__ void cuda_update_weights(int num_features, double* weight_vector, uint8_t* features, uint8_t label,
    double* probabilities, double* max, double* total) {
  double probability = probabilities[blockIdx.x];

  probability /= *total;

  for (int i = threadIdx.x; i < num_features; i += blockDim.x) {
    int offset = blockIdx.x * num_features + i;
    
    double y = (blockIdx.x == label) ? 1 : 0;
    weight_vector[offset] += (y - probability) * features[i];
  }
}

int greatest_pow2(int n) {
  return (int)pow(2, (int)log2((float)n));
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

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // For each training point:
  // 1. Calculate gradient.
  // 2. Update weight vector.
  // Continue through entire dataset.
  int powerof2 = greatest_pow2(ds->nFeatures);//(int)pow(2, (int)log2((float)ds-> n));
  int shared_mem_size = powerof2 * sizeof(double);

  for (int i = 0; i < ds->train_size; i++) {
    cuda_compute_probabilities<<<ds->nClasses, powerof2, shared_mem_size>>>(probabilities, d_weight_vectors, &d_train_set[i * ds->nFeatures], ds->nFeatures);
    cuda_find_max<<<1,1>>>(ds->nClasses, probabilities, max, total);
    cuda_update_weights<<<ds->nClasses, min(1024, ds->nFeatures)>>>(ds->nFeatures, d_weight_vectors, &d_train_set[i * ds->nFeatures],
      ds->train_labels[i], probabilities, max, total);
 }

  cudaEventRecord(stop);

  for (int i = 0; i < ds->nClasses; i++) {
    int offset = i * ds->nFeatures;
    cudaMemcpy(weight_vectors[i], d_weight_vectors + offset,
      ds->nFeatures * sizeof(double), cudaMemcpyDeviceToHost);
  }

  float duration;
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&duration, start, stop);
  printf("train duration: %f ms\n", duration);

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
