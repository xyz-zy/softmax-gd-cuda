#include <iostream> // cout & endl
#include <getopt.h>

#include <cstdlib> // atoi(), atof()
#include <string> // stoi();
#include <fstream> // read and write from/to files
#include <sstream>
#include <iterator>


#include <cmath> // pow()


uint8_t dataset[42000][784];
uint8_t labels[42000];

int main(int argc, char *argv[]) {

  // 42001 rows × 785 columns
  // First row represents column headers
  // First column represents image labels
  std::ifstream input_file;
  input_file.open("../train.csv");

  std::string line;
  getline(input_file, line);

  //printf("%s\n", line.c_str());

  for (int i = 0; i < 42; i++) {
    for (int j = 0; j < 784; j++) {
      getline(input_file, line, ',');

      if (j == 0) {
        labels[i] = stoi(line);
      } else {
        dataset[i][j-1] = stoi(line);
      }
    }
    getline(input_file, line);
    dataset[i][783] = stoi(line);
  }

  input_file.close();

  // for (int i = 0; i < 42; i++) {
  //   for (int j = 0; j < 784; j++) {
  //     printf("%d ", dataset[i][j]);
  //   }
  //   printf("\n");
  // }
}
