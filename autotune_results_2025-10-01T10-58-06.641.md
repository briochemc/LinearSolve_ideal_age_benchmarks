## LinearSolve.jl Autotune Benchmark Results

### Performance Summary by Size Range
#### Recommendations for Float64

| Size Range | Best Algorithm |
|------------|----------------|
| tiny (5-20) | RFLUFactorization |
| small (20-100) | RFLUFactorization |
| medium (100-300) | RFLUFactorization |
| large (300-1000) | AppleAccelerateLUFactorization |
| big (1000+) | OpenBLASLUFactorization |


### Detailed Results
#### Results for Float64

##### Summary Statistics

| Algorithm | Avg GFLOPs | Std Dev | Success/Total |
|-----------|------------|---------|---------------|
| AppleAccelerateLUFactorization | 32.88 | 37.86 | 19/19 |
| OpenBLASLUFactorization | 30.74 | 35.65 | 19/19 |
| LUFactorization | 25.94 | 30.37 | 19/19 |
| RFLUFactorization | 23.29 | 14.93 | 19/19 |
| GenericLUFactorization | 3.25 | 1.42 | 19/19 |
| SimpleLUFactorization | 3.18 | 1.38 | 19/19 |

<details>
<summary>Raw Performance Data</summary>

##### AppleAccelerateLUFactorization

| Matrix Size | GFLOPs | Status |
|-------------|--------|--------|
| 5 | 0.096 | ✅ Success |
| 10 | 0.467 | ✅ Success |
| 15 | 1.014 | ✅ Success |
| 20 | 1.662 | ✅ Success |
| 40 | 1.035 | ✅ Success |
| 60 | 2.098 | ✅ Success |
| 80 | 4.785 | ✅ Success |
| 100 | 5.202 | ✅ Success |
| 150 | 10.389 | ✅ Success |
| 200 | 9.756 | ✅ Success |
| 250 | 24.882 | ✅ Success |
| 300 | 35.407 | ✅ Success |
| 400 | 48.280 | ✅ Success |
| 500 | 57.983 | ✅ Success |
| 600 | 36.868 | ✅ Success |
| 700 | 86.305 | ✅ Success |
| 800 | 99.555 | ✅ Success |
| 900 | 97.431 | ✅ Success |
| 1000 | 101.514 | ✅ Success |

##### GenericLUFactorization

| Matrix Size | GFLOPs | Status |
|-------------|--------|--------|
| 5 | 0.181 | ✅ Success |
| 10 | 0.778 | ✅ Success |
| 15 | 1.250 | ✅ Success |
| 20 | 1.638 | ✅ Success |
| 40 | 2.681 | ✅ Success |
| 60 | 3.213 | ✅ Success |
| 80 | 3.371 | ✅ Success |
| 100 | 3.507 | ✅ Success |
| 150 | 3.282 | ✅ Success |
| 200 | 3.772 | ✅ Success |
| 250 | 3.709 | ✅ Success |
| 300 | 3.840 | ✅ Success |
| 400 | 4.359 | ✅ Success |
| 500 | 4.229 | ✅ Success |
| 600 | 5.227 | ✅ Success |
| 700 | 5.103 | ✅ Success |
| 800 | 2.591 | ✅ Success |
| 900 | 4.466 | ✅ Success |
| 1000 | 4.557 | ✅ Success |

##### LUFactorization

| Matrix Size | GFLOPs | Status |
|-------------|--------|--------|
| 5 | 0.119 | ✅ Success |
| 10 | 0.708 | ✅ Success |
| 15 | 0.922 | ✅ Success |
| 20 | 1.209 | ✅ Success |
| 40 | 2.819 | ✅ Success |
| 60 | 4.909 | ✅ Success |
| 80 | 7.532 | ✅ Success |
| 100 | 8.952 | ✅ Success |
| 150 | 11.108 | ✅ Success |
| 200 | 8.487 | ✅ Success |
| 250 | 12.593 | ✅ Success |
| 300 | 12.811 | ✅ Success |
| 400 | 28.522 | ✅ Success |
| 500 | 39.532 | ✅ Success |
| 600 | 39.033 | ✅ Success |
| 700 | 65.702 | ✅ Success |
| 800 | 83.371 | ✅ Success |
| 900 | 80.606 | ✅ Success |
| 1000 | 83.854 | ✅ Success |

##### OpenBLASLUFactorization

| Matrix Size | GFLOPs | Status |
|-------------|--------|--------|
| 5 | 0.117 | ✅ Success |
| 10 | 0.516 | ✅ Success |
| 15 | 0.944 | ✅ Success |
| 20 | 1.279 | ✅ Success |
| 40 | 3.627 | ✅ Success |
| 60 | 5.402 | ✅ Success |
| 80 | 8.361 | ✅ Success |
| 100 | 9.334 | ✅ Success |
| 150 | 13.050 | ✅ Success |
| 200 | 9.163 | ✅ Success |
| 250 | 14.736 | ✅ Success |
| 300 | 18.670 | ✅ Success |
| 400 | 32.565 | ✅ Success |
| 500 | 45.726 | ✅ Success |
| 600 | 61.043 | ✅ Success |
| 700 | 77.652 | ✅ Success |
| 800 | 78.352 | ✅ Success |
| 900 | 96.841 | ✅ Success |
| 1000 | 106.683 | ✅ Success |

##### RFLUFactorization

| Matrix Size | GFLOPs | Status |
|-------------|--------|--------|
| 5 | 0.167 | ✅ Success |
| 10 | 0.748 | ✅ Success |
| 15 | 1.849 | ✅ Success |
| 20 | 2.685 | ✅ Success |
| 40 | 7.302 | ✅ Success |
| 60 | 12.427 | ✅ Success |
| 80 | 16.393 | ✅ Success |
| 100 | 19.959 | ✅ Success |
| 150 | 26.358 | ✅ Success |
| 200 | 29.130 | ✅ Success |
| 250 | 28.999 | ✅ Success |
| 300 | 31.361 | ✅ Success |
| 400 | 36.084 | ✅ Success |
| 500 | 35.526 | ✅ Success |
| 600 | 37.304 | ✅ Success |
| 700 | 36.570 | ✅ Success |
| 800 | 40.698 | ✅ Success |
| 900 | 39.403 | ✅ Success |
| 1000 | 39.455 | ✅ Success |

##### SimpleLUFactorization

| Matrix Size | GFLOPs | Status |
|-------------|--------|--------|
| 5 | 0.184 | ✅ Success |
| 10 | 0.644 | ✅ Success |
| 15 | 1.255 | ✅ Success |
| 20 | 1.601 | ✅ Success |
| 40 | 2.574 | ✅ Success |
| 60 | 3.017 | ✅ Success |
| 80 | 3.228 | ✅ Success |
| 100 | 3.327 | ✅ Success |
| 150 | 3.579 | ✅ Success |
| 200 | 3.663 | ✅ Success |
| 250 | 3.710 | ✅ Success |
| 300 | 3.832 | ✅ Success |
| 400 | 3.739 | ✅ Success |
| 500 | 4.402 | ✅ Success |
| 600 | 4.747 | ✅ Success |
| 700 | 4.793 | ✅ Success |
| 800 | 4.999 | ✅ Success |
| 900 | 4.203 | ✅ Success |
| 1000 | 2.974 | ✅ Success |

</details>


### System Information
- **Julia Version**: 1.11.7
- **OS**: macOS (Darwin)
- **Architecture**: x86_64
- **CPU Model**: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
- **CPU Speed**: 2600 MHz
- **Cores**: 6
- **Threads**: 1
- **BLAS**: lbt
- **MKL Available**: false
- **Apple Accelerate Available**: true
- **CUDA Available**: false
- **Metal Available**: false

---
*Generated automatically by LinearSolveAutotune.jl*
