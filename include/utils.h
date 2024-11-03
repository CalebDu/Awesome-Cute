#pragma once

#include "cutlass/numeric_types.h"
#include <cstdarg>
#include <iostream>
#include <stdlib.h>

void printf_fail(const char *fmt, ...) {
  int red = 31;
  int def = 39;

  printf("\033[%dm", red);

  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  printf("\033[%dm", def);
}

void printf_pass(const char *fmt, ...) {
  int red = 32;
  int def = 39;

  printf("\033[%dm", red);

  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  printf("\033[%dm", def);
}

template <typename T> struct UnderlyingType {
  using type = T;
};
template <> struct UnderlyingType<cutlass::half_t> {
  using type = half;
};

inline
float compute_tflops(float flop, float ms) {
  float tflops = flop * 1e-9 / ms;
  return tflops;
}
