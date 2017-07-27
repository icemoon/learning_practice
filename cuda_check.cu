#include <iostream>

using namespace std;

void _cudaCheck(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(err)
            << file << line << endl;
        exit(-1);

    }

}
#define cudaCheck(ans) { _cudaCheck((ans), __FILE__, __LINE__);  }


__global__  void add(int *a, int *b, int *c)
{
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];

}


int main()
{
    const int N = 16;

    int a[N] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    int b[N] = {16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};

    const size_t sz = N * sizeof(int);
    int *da;
    cudaCheck(cudaMalloc(&da, sz));
    int *db;
    cudaCheck(cudaMalloc(&db, sz));
    int *dc;
    cudaCheck(cudaMalloc(&dc, sz));

    cudaCheck(cudaMemcpy(da, a, sz, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(db, b, sz, cudaMemcpyHostToDevice));

    add<<<1, N>>>(da, db, dc);
    cudaCheck(cudaGetLastError());

    int c[N];
    cudaCheck(cudaMemcpy(c, dc, sz, cudaMemcpyDeviceToHost));

    cudaCheck(cudaFree(da));
    cudaCheck(cudaFree(db));
    cudaCheck(cudaFree(dc));

    for (unsigned int i = 0 ; i < N; ++i) {
        cout << c[i] << " ";

    }
    cout << endl;
    return 0;

}
