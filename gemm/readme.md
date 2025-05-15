
# Gemm
This folder is Gemm implementation.

## Execute
``` shell
../bin/gemm_mutlistage m n k
../bin/gemm_streamk m n k
../bin/gemm_ws_sm8x m n k
python ./marlin_gemm/marlin_profiling.py
./bin/gemm_ws_sm90a m n k swizzle
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

### Marlin W4A16 Gemm
|   |   |   |
|---|---|---|
|shape(m_n_k_group)|marlin_cute|marlin_official|
|7B_1_12288_4096_-1|15.057 us|14.629 us|
|7B_16_12288_4096_-1|15.941 us|15.291 us|
|7B_1_12288_4096_128|15.744 us|14.921 us|
|7B_16_12288_4096_128|16.550 us|15.637 us|
|7B_1_4096_4096_-1|8.359 us|8.189 us|
|7B_16_4096_4096_-1|9.035 us|8.936 us|
|7B_1_4096_4096_128|8.714 us|8.431 us|
|7B_16_4096_4096_128|9.393 us|9.113 us|
|7B_1_21504_4096_-1|23.983 us|23.464 us|
|7B_16_21504_4096_-1|24.858 us|24.171 us|
|7B_1_21504_4096_128|25.268 us|24.068 us|
|7B_16_21504_4096_128|26.104 us|24.861 us|
|7B_1_4096_10752_-1|14.108 us|13.845 us|
|7B_16_4096_10752_-1|14.916 us|14.617 us|
|7B_1_4096_10752_128|14.872 us|14.172 us|
|7B_16_4096_10752_128|15.714 us|15.224 us|

### Warp Spcialization Gemm for Hopper
experiment performance in H100 PCIE 80GB, cuda_12.8.r12.8


|                      | 2048x2048x2048                                      | 4096x4095x4096                                      | 8192x8192x8192                                       | 备注                                                                         |
|----------------------|-----------------------------------------------------|-----------------------------------------------------|------------------------------------------------------|------------------------------------------------------------------------------|
| cublas gemm          | ncu: 49788cycle      stream: 626tflops  0.027419 ms | ncu: 282381cycle      stream:775tflops 0.177158 ms  | ncu: 2270766cycle      stream:716tflops 1.533956 ms  | stream   测出cublas的latency更少算力更高，但是ncu 显示的cublas 的cycle数更多 |
| cutlass_ws gemm      | ncu: 51681cycle      stream: 521tflops 0.032957 ms  | ncu: 325350cycle      stream: 596tflops 0.230587 ms | ncu: 2825726cycle      stream:516tflops 1.815191 ms  | cutlass ws 还是采用data   parallel的策略而非persistent                       |
| cutlass_ws_coop gemm | ncu: 47620cycle      stream:566tflops  0.030308 ms  | ncu: 293475cycle      stream:679tflops 0.202164 ms  | ncu: 2291953cycle      stream:598tflops 1.837878 ms  |                                                                              |
| cutlass_ws_pipo gemm | ncu: 46299cycle      stream:564tflops 0.030417 ms   | ncu: 279205cycle      stream:695tflops 0.197744 ms  | ncu: 2237864cycle      stream:608tflops 1.805988 ms  |                                                                              |
| my_ws gemm           | ncu: 44868cycle      stream: 628tflops 0.027348 ms  | ncu: 291775cycle      stream: 694tflops 0.197875 ms | ncu: 2195027cycle      stream:605tflops 1.815191 ms  |                                                                              |
| my_ws_coop gemm      | ncu: 43795cycle      stream: 629tflops 0.027309 ms  | ncu: 287676cycle      stream:683tflops 0.201130 ms  | ncu: 2173536cycle      stream: 626tflops 1.753922 ms |                                                                              |
| my_ws_pipo gemm      | ncu: 42447cycle      stream: 629tflops  0.026972 ms | ncu: 275501cycle      stream: 710tflops 0.193317 ms | ncu: 2129913cycle      stream:638tflops 1.722661 ms  |                                                                              |