#include <torch/extension.h>


using SoftmaxFn = void(
    const float* in,
    float* out,
    int N, int H
);

SoftmaxFn softmax_v1;
SoftmaxFn softmax_v2;
SoftmaxFn softmax_v3a;
SoftmaxFn softmax_v3b;
SoftmaxFn softmax_v4;
SoftmaxFn softmax_v5;

template <SoftmaxFn softmax_fn>
torch::Tensor softmax(torch::Tensor input)
{
    const int N = input.size(0);
    const int H = input.size(1);
    auto output = torch::empty_like(input);
    softmax_fn(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, H
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_v1", &softmax<softmax_v1>, "Softmax v1");
    m.def("softmax_v2", &softmax<softmax_v2>, "Softmax v2");
    m.def("softmax_v3a", &softmax<softmax_v3a>, "Softmax v3a");
    m.def("softmax_v3b", &softmax<softmax_v3b>, "Softmax v3b");
    m.def("softmax_v4", &softmax<softmax_v4>, "Softmax v4");
    m.def("softmax_v5", &softmax<softmax_v5>, "Softmax v5");
}
