#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include <cassert>
#include <cstdio>

using namespace std;

void log_values_print(string in_file, hsize_t num_lines) {
    string out_file = "log_" + in_file;
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
    
    uint64_t buffer_size = 10000;

    std::string buf = "";
    uint64_t count = 0;

    vector<uint64_t> tuple(dim, 0); 

    FILE *out_file_ptr = fopen(out_file.c_str(), "w");

    for(uint64_t i = 0; i < num_lines; i++) {
        if(count == buffer_size) {
            count = 0;
            fprintf(out_file_ptr, buf.c_str());
            buf = "";
        }

        for(int j = 0; j < dim; j++) {
            iffstream >> tuple[j];
        }
        double val;
        iffstream >> val;
        buf << 
    }

    fclose(out_file_ptr);

}

int main(int argc, char** argv) {
}

