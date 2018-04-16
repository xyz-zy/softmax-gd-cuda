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

uint8_t train_set[TRAIN_SIZE][784];
uint8_t train_labels[TRAIN_SIZE];

uint8_t test_set[TEST_SIZE][784];
uint8_t test_labels[TEST_SIZE];

void load_data() {
  // 42001 rows × 785 columns
  // First row represents column headers
  // First column represents image labels
  std::ifstream input_file;
  input_file.open("train.csv");

  std::string line;
  getline(input_file, line);

  //printf("%s\n", line.c_str());

  for (int i = 0; i < TRAIN_SIZE; i++) {
    for (int j = 0; j < 784; j++) {
      getline(input_file, line, ',');

      if (j == 0) {
        train_labels[i] = stoi(line);
      } else {
        train_set[i][j-1] = stoi(line);
      }
    }
    getline(input_file, line);
    train_set[i][783] = stoi(line);
  }

  for (int i = 0; i < TEST_SIZE; i++) {
    for (int j = 0; j < 784; j++) {
      getline(input_file, line, ',');

      if (j == 0) {
        test_labels[i] = stoi(line);
      } else {
        test_set[i][j-1] = stoi(line);
      }
    }
    getline(input_file, line);
    test_set[i][783] = stoi(line);
  }

  input_file.close();
}

double* generate_weight_vector(int size) {
  double* w = new double[size];

  std::srand(std::time(nullptr));

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

void update_gradients(double** weight_vectors, uint8_t* features, uint8_t label) {
  double* probabilities = new double[10];
  double min_product = DBL_MAX;

  for (int i = 0; i < 10; i++) {
    double product = inner_product(784, weight_vectors[i], features);
    probabilities[i] = product;
    if (product < min_product) {
      min_product = product;
    }
  } 

  // Exponentiate and calculate sum.
  double sum = 0;

  for (int i = 0; i < 10; i++) {
    sum += std::exp(probabilities[i] - min_product);
  }

  // Calculate probabilities.
  for (int i = 0; i < 10; i++) {
    probabilities[i] /= sum;
  }

  // Update weight vectors.
  for (int i = 0; i < 10; i++) {
    int y = (i == label) ? 1 : 0;
    for (int j = 0; j < 784; j++) {
      weight_vectors[i][j] += (y - probabilities[i]) * features[j];
    }
  }

}

void sgd() {
  // Generate random weight vector (784).

  double** weight_vectors = generate_k_weight_vectors(10, 784);

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 784; j++) {
      //printf("%f ", weight[i]);
      if (weight_vectors[i][j] <= 0 || weight_vectors[i][j] >= 1) {
        printf("INVALID WEIGHT");
      }
    }
  }

  // For each training point:
  // 1. Generate prediction.
  // 2. Calculate gradient.
  // 3. Update weight vector.
  // Continue through entire dataset.
  for (int i = 0; i < TRAIN_SIZE; i++) {
    //uint8_t prediction = predict(10, 784, weight_vectors, train_set[i]);
    update_gradients(weight_vectors, train_set[i], train_labels[i]);
  }
}

void test() {
  // For each training point, generate prediction.

  // Compute total accuracy.
}


int main(int argc, char *argv[]) {
  load_data();
  sgd();
  // TODO: Antony Yun
  test();

  // for (int i = 0; i < 42; i++) {
  //   for (int j = 0; j < 784; j++) {
  //     printf("%d ", dataset[i][j]);
  //   }
  //   printf("\n");
  // }
}
