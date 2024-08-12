# 记录一下CUDA的学习

**检查显卡的各项参数**: 

以我的为例，我的显卡参数如下：

```bash
Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce RTX 3080"
  Total amount of global memory:                 10240 MBytes (10736893952 bytes)
  GPU Max Clock rate:                            1710 MHz (1.71 GHz)
  L2 Cache Size:                                 5242880 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a block size (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size  (x,y,z): (2147483647, 65535, 65535)
```

