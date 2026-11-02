#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

template <typename T>
T dot(const std::vector<T> &a, const std::vector<T> &b) {
    T s{};
    for (std::size_t i = 0; i < a.size(); i++) s += a[i] * b[i];
    return s;
}

int main() {
    std::vector<long> a{3, 1, 4, 1, 5, 9, 2, 6};
    std::vector<long> b(a.rbegin(), a.rend());
    std::sort(a.begin(), a.end());
    std::map<std::string, long> m{{"dot", dot(a, b)}, {"max", a.back()}};
    for (const auto &[k, v] : m) std::printf("%s=%ld\n", k.c_str(), v);
    return 0;
}
