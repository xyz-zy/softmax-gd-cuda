#include <iostream> // cout & endl
#include <getopt.h>

#include <cstdlib> // atoi(), atof()
#include <string> // stoi();
#include <fstream> // read and write from/to files
#include <sstream>
#include <iterator>


#include <ctime> // time()
#include <limits> // FLT_MAX, DBL_MAX
#include <cmath> // pow()


uint8_t train_set[40000][784];
uint8_t train_labels[40000];

uint8_t test_set[2000][784];
uint8_t test_labels[2000];

double* generate_weight_vector(int size) {
  double* w = new double[size];

  std::srand(std::time(nullptr));

  for(int i = 0; i < size; i++) {
    w[i] = (double) std::rand() / RAND_MAX;
  }
  return w;
}

void sgd() {
  // Generate random weight vector (784).

  double* weight = generate_weight_vector(784);

  for (int i = 0; i < 784; i++) {
    //printf("%f ", weight[i]);
    if (weight[i] <= 0 || weight[i] >= 1) {
      printf("INVALID WEIGHT");
    }
  }

  // For each training point:
  // 1. Generate prediction.
  // 2. Calculate gradient.
  // 3. Update weight vector.
  // Continue through entire dataset.
}

void test() {
  // For each training point, generate prediction.

  // Compute total accuracy.
}


int main(int argc, char *argv[]) {

  // 42001 rows × 785 columns
  // First row represents column headers
  // First column represents image labels
  std::ifstream input_file;
  input_file.open("../train.csv");

  std::string line;
  getline(input_file, line);

  //printf("%s\n", line.c_str());

  for (int i = 0; i < 40000; i++) {
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

  for (int i = 0; i < 2000; i++) {
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

  sgd();

  // for (int i = 0; i < 42; i++) {
  //   for (int j = 0; j < 784; j++) {
  //     printf("%d ", dataset[i][j]);
  //   }
  //   printf("\n");
  // }
}
