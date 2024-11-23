
# Gemm
This folder is Gemm implementation.

## Execute
``` shell
../bin/gemm_mutlistage m n k
../bin/gemm_streamk m n k
../bin/gemm_ws m n k
```

## Performance
experiment performance in rtx 4090, cuda_12.1.r12.1.
### Multistage Gemm
|test case|cublas|cute: gemm_multistage_128\*128\*32_stage3_nocheck|cute: gemm_multistage_128\*256\*32_stage3_nocheck|cute: gemm_multistage_128\*128\*32_stage3_check_bound|cute: gemm_multistage_128\*256\*32_stage3_check_bound|
|---|---|---|---|---|---|
|mnk(4096,4096,4096)|242tflops/0.566ms|258tflops/0.532ms|272tflops/0.504ms|234tflops/0.585ms|148tflops/0.922ms|
|mnk(2048,2048,2048)|210tflops/0.081ms|258tflops/0.066ms|273tflops/0.062ms|232tflops/0.738ms|141tflops/0.121ms|
|mnk(8192, 8192,8192)|235tflops/4.665ms|208tflops/5.27ns|257tflops/4.265ms|189tflops/5.811ms|144tflops/7.595ms|

### StreamK Gemm
|test case：amount of tiles not divisible by SM|cublas|cute: gemm_multistage|cute: gemm_streamk_1sk_dp|cute: gemm_streamk_2sk_dp_128\*256\*32_stage3|cutlass:example/47_ampere_gemm_universal_streamk|
|---|---|---|---|---|---|
|mnk(4096,4352,4096)|235tflops/0.619ms|249tflops/0.585ms(128\*128\*32_stage3)|257tflops/0.566ms(_128\*256\*32_stage3)|271tflops/0.538|270tflops/0.553ms(default load-balancing)|
|mnk(4096,4352,10240)|235tflops/1.545ms|239tflops/1.521ms(128\*256\*32_stage3)|258tflops/1.414ms(_128\*256\*32_stage3)|265tflops/1.373ms|263tflops/1.384ms(default load-balancing)|
|mnk(1152,4352,4096)|219tflops/0.186ms|218tflops/0.187ms(128\*128\*32_stage3)|255tflops/0.160ms(gemm_streamk_1sk_dp_128\*128\*32_stage3)|268tflops/0.153ms|272tflops/1.504ms(default load-balancing)|

### Naive Warp Spcialization Gemm
  
||cublas|cute: gemm_multistage_128*256*32_stage3|cute: gemm_ws_producer32_128*256*32_stage3|cute: gemm_ws_producer64_128*256*32_stage3|cute: gemm_ws_producer128_128*256*32_stage3|
|---|---|---|---|---|---|
|mnk(2048,2048,2048)|218tflops/0.078ms|275tflops/0.062ms|235tflops/0.072ms|246tflops/0.069ms|252tflops/0.068ms|
|mnk(4096,4096,4096)|260tflops/0.525ms|289tflops/0.475ms|247tflops/0.556ms|259tflops/0.528ms|267tflops/0.514ms|
|mnk(8192,8192,8192)|252tflops/4.353ms|268tflops/4.101ms|211tflops/5.206ms|247tflops/4.436ms|255tflops/4.304ms|

---