#include <iostream>
#include <memory>
#include <vector>

using namespace std;

template<typename T>
class __attribute__((visibility("hidden"))) Buffer {
    unique_ptr<T[]> managed_ptr;
    T* ptr;
    uint64_t dim0;
    uint64_t dim1;

public:
    vector<uint64_t> shape;

    Buffer(Buffer&& other)
          :
            managed_ptr(std::move(other.managed_ptr)),
            ptr(other.ptr),
            dim0(other.dim0),
            dim1(other.dim1),
            shape(std::move(other.shape))
    {}
    Buffer& operator=(const Buffer& other) = default;

    Buffer(initializer_list<uint64_t> args) {
        uint64_t buffer_size = 1;
        for(uint64_t i : args) {
            buffer_size *= i;
            shape.push_back(i);
        }

        if(args.size() == 2) {
            dim0 = shape[0];
            dim1 = shape[1];
        }

        managed_ptr.reset(new T[buffer_size]);
        ptr = managed_ptr.get();
    }

    Buffer(initializer_list<uint64_t> args, T* ptr) {
        for(uint64_t i : args) {
            shape.push_back(i);
        }

        if(args.size() == 2) {
            dim0 = shape[0];
            dim1 = shape[1];
        }

        this->ptr = ptr;
    }

    T* operator()() {
        return ptr;
    }

    T* operator()(uint64_t offset) {
        return ptr + offset;
    }

    // Assumes that this array is a row-major matrix 
    T* operator()(uint64_t off_x, uint64_t off_y) {
        return ptr + (dim1 * off_x) + off_y;
    }

    T& operator[](uint64_t offset) {
        return ptr[offset];
    }

    ~Buffer() {
    }
};

int main(int argc, char** argv) {
    unique_ptr<Buffer<double>> x;
    vector<Buffer<double>> vec;

    for(uint64_t i = 0; i < 10; i++) {
      vec.emplace_back(std::initializer_list<uint64_t>({500, 500}));
    }
    x.reset(new Buffer<double> ({1000, 1000}));
    cout << "Hello world!" << endl;
}