#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstdio>

using namespace std;

void log_values_print(string in_file) {
    string out_file = in_file + "_log";
    std::string line, token;
    std::ifstream firstline_stream(in_file, std::ifstream::in);
    std::ifstream iffstream(in_file, std::ifstream::in);

    // Read the first line and count the number of tokens 
    std::getline(firstline_stream, line); 
    std::istringstream is( line );
    
    int count = 0;
    while ( std::getline( is, token, ' ')) {
        ++count;
    }
    int dim = count - 1;

    uint64_t buffer_size = 100;

    std::string buf = "";
    count = 0;

    FILE *out_file_ptr = fopen(out_file.c_str(), "w");

    while(std::getline(iffstream, line)) {
        if(count == buffer_size) {
            count = 0;
            fprintf(out_file_ptr, "%s", buf.c_str());    
            buf = "";
        }

        // Split line into tokens separated by spaces
        std::istringstream is( line );
        int i = 0;
        while ( std::getline( is, token, ' ')) {
            if(i < dim) {
                buf += token + " "; 
            }
            else {
                buf += to_string(log(stod(token) + 1)) + "\n";
            }
            i++;
        }
        count++;
    }

    if(count > 0) {
        fprintf(out_file_ptr, "%s", buf.c_str());
    }
    fclose(out_file_ptr);
}

int main(int argc, char** argv) {
    if(argc != 2) {
        cout << "Usage: ./log_values <input_file>" << endl;
        exit(1);
    }

    string in_file(argv[1]);
    log_values_print(in_file);
}

