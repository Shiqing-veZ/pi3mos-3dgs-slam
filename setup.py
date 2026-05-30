import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))
EIGEN = osp.join(ROOT, "thirdparty/eigen-3.4.0")
GLM = osp.join(ROOT, "thirdparty/diff_gaussian_rasterization/third_party")



setup(
    name='dpvo',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('cuda_corr',
            sources=['dpvo/altcorr/correlation.cpp', 'dpvo/altcorr/correlation_kernel.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            }),
        CUDAExtension('cuda_ba',
            sources=['dpvo/fastba/ba.cpp', 'dpvo/fastba/ba_cuda.cu', 'dpvo/fastba/block_e.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            },
            include_dirs=[
                EIGEN]
            ),
        CUDAExtension('lietorch_backends', 
            include_dirs=[
                osp.join(ROOT, 'dpvo/lietorch/include'),
                EIGEN],
            sources=[
                'dpvo/lietorch/src/lietorch.cpp', 
                'dpvo/lietorch/src/lietorch_gpu.cu',
                'dpvo/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3'],}),
        CUDAExtension(
            'thirdparty.simple_knn._C',
            sources=[
                'thirdparty/simple_knn/spatial.cu',
                'thirdparty/simple_knn/simple_knn.cu',
                'thirdparty/simple_knn/ext.cpp',
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']},
        ),
        CUDAExtension(
            'thirdparty.diff_gaussian_rasterization._C',
            sources=[
                'thirdparty/diff_gaussian_rasterization/cuda_rasterizer/rasterizer_impl.cu',
                'thirdparty/diff_gaussian_rasterization/cuda_rasterizer/forward.cu',
                'thirdparty/diff_gaussian_rasterization/cuda_rasterizer/backward.cu',
                'thirdparty/diff_gaussian_rasterization/rasterize_points.cu',
                'thirdparty/diff_gaussian_rasterization/ext.cpp',
            ],
            include_dirs=[GLM],
            extra_compile_args={'nvcc': ['-O3', '-I' + GLM], 'cxx': ['-O3']},
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
