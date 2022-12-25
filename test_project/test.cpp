#include <stddef.h>
#include <stdio.h>
#include <execution>
#include <algorithm>
#include <chrono>
#include <random>
#include <ratio>
#include <vector>
#include "omp.h"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::milli;
using std::random_device;
using std::sort;
using std::vector;

const size_t testSize = 100'000'000;
const int iterationCount = 5;

void print_results(const char *const tag, const vector<double>& sorted,
                   high_resolution_clock::time_point startTime,
                   high_resolution_clock::time_point endTime) {
  printf("%s: Lowest: %g Highest: %g Time: %fms\n", tag, sorted.front(),
         sorted.back(),
         duration_cast<duration<double, milli>>(endTime - startTime).count());
}

int main() {
  random_device rd;

  // generate some random doubles:
  printf("Testing with %zu doubles...\n", testSize);
  vector<double> doubles(testSize);
  for (auto& d : doubles) {
    d = static_cast<double>(rd());
  }

  // time how long it takes to sort them:
  for (int i = 0; i < iterationCount; ++i)
  {
    vector<double> sorted(doubles);
    const auto startTime = high_resolution_clock::now();
    sort(std::execution::par_unseq, sorted.begin(), sorted.end()); 
    //sort(sorted.begin(), sorted.end());
    const auto endTime = high_resolution_clock::now();
    print_results("Serial", sorted, startTime, endTime);
  }
}