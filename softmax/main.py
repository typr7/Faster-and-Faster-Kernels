import torch
import torch.utils.cpp_extension
from triton.testing import do_bench
import statistics

def benchmark(f, *args, **kwargs):
    samples = [do_bench(lambda: f(*args, **kwargs), return_mode="median")
               for _ in range(7)]
    return statistics.median(samples)

module = torch.utils.cpp_extension.load(
    'module',
    sources=[
        'softmax.cpp',
        'softmax_v1.cu',
        'softmax_v2.cu',
        'softmax_v3a.cu',
        'softmax_v3b.cu',
        'softmax_v4.cu',
        'softmax_v5.cu'
    ],
    extra_cuda_cflags=[
        '-O3',
        '-lineinfo',
        '-Xptxas=-v'
    ],
    verbose=True
)

input = torch.randn((4096, 4096), dtype=torch.float32).cuda()

output_ref = torch.softmax(input, dim=1)
output_v1  = module.softmax_v1(input)
output_v2  = module.softmax_v2(input)
output_v3a = module.softmax_v3a(input)
output_v3b = module.softmax_v3b(input)
output_v4  = module.softmax_v4(input)
output_v5  = module.softmax_v5(input)

torch.testing.assert_close(output_v1, output_ref)
torch.testing.assert_close(output_v2, output_ref)
torch.testing.assert_close(output_v3a, output_ref)
torch.testing.assert_close(output_v3b, output_ref)
torch.testing.assert_close(output_v4, output_ref)
torch.testing.assert_close(output_v5, output_ref)

print(f'torch.softmax: {benchmark(torch.softmax, input, dim=1)}')
print(f'v1: {benchmark(module.softmax_v1, input)}')
print(f'v2: {benchmark(module.softmax_v2, input)}')
print(f'v3a: {benchmark(module.softmax_v3a, input)}')
print(f'v3b: {benchmark(module.softmax_v3b, input)}')
print(f'v4: {benchmark(module.softmax_v4, input)}')
print(f'v5: {benchmark(module.softmax_v5, input)}')
