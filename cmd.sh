ncu --nvtx --call-stack --set full --launch-count 10 -f --export ./test_kernels.ncu-rep ./test_kernels
ncu-ui ./test_kernels.ncu-rep
