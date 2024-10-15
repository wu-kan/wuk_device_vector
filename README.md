# wuk_device_vector

为方便日后使用，基于 `thrust::device_vector` 提供两个常用函数：

- `equal` 当 `lhs` 和 `rhs` 的元素数量相等，且对应位置的元素之差的绝对值小于 `eps` 时返回 `true`，否则返回 `false`。
- `generate` 在 `v` 上以种子 `s` 在 $[a,b)$ 上默认产生均匀分布。

```cpp
namespace wuk {

using thrust::device_vector;

using thrust::raw_pointer_cast;

template <typename T>
bool equal(const device_vector<T> &lhs, const device_vector<T> &rhs, T eps);

template <typename T,
          typename Tdistribution = thrust::uniform_real_distribution<T>>
void generate(device_vector<T> &v, T a, T b,
              int s = thrust::default_random_engine::default_seed);

}
```
