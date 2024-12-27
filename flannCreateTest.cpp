#include <fstream>
#include <vector>

int main() {
    std::ofstream ofs("vectors.bin", std::ios::binary);
    const size_t DIM = 384;
    const size_t N = 100;
    for (size_t i = 0; i < N; i++) {
        std::vector<float> vec(DIM);
        for (size_t j = 0; j < DIM; j++) {
            vec[j] = float(i) + float(j)*0.001f; // some dummy data
        }
        ofs.write(reinterpret_cast<const char*>(vec.data()), DIM * sizeof(float));
    }
    return 0;
}
