#include <arm_neon.h>

#define SIMD_WIDTH_FLOAT 4
#define SIMD_WIDTH_HALF 8

// Store and Load macros for float and half
#define SIMD_STORE_FLOAT(a, d) vst1q_f32(a, d)
#define SIMD_LOAD_FLOAT(x) vld1q_f32(x)
#define SIMD_STORE_HALF(a, d) vst1_f16((__fp16*)(a), vcvt_f16_f32(d))
#define SIMD_LOAD_HALF(x) vcvt_f32_f16(vld1_f16((const __fp16*)(x)))

#define SIMD_STORE2(x, d, h) ((h) ? SIMD_STORE_HALF(x, d) : SIMD_STORE_FLOAT(x, d))
#define SIMD_LOAD2(x, h) ((h) ? SIMD_LOAD_HALF(x) : SIMD_LOAD_FLOAT(x))
#define SIMD_WIDTH(h) ((h) ? SIMD_WIDTH_HALF : SIMD_WIDTH_FLOAT)

// Operation macros for float and half
#define SIMD_SET_FLOAT(x) vdupq_n_f32(x)
#define SIMD_SET_HALF(x) vcvt_f32_f16(vdup_n_f16(x))
#define SIMD_ADD_FLOAT(x, y) vaddq_f32(x, y)
#define SIMD_ADD_HALF(x, y) vcvt_f32_f16(vadd_f16(vcvt_f16_f32(x), vcvt_f16_f32(y)))
#define SIMD_MUL_FLOAT(x, y) vmulq_f32(x, y)
#define SIMD_MUL_HALF(x, y) vcvt_f32_f16(vmul_f16(vcvt_f16_f32(x), vcvt_f16_f32(y)))
#define SIMD_FMA_FLOAT(x, y, c) vmlaq_f32(c, x, y)
#define SIMD_FMA_HALF(x, y, c) vcvt_f32_f16(vmla_f16(vcvt_f16_f32(c), vcvt_f16_f32(x), vcvt_f16_f32(y)))
#define SIMD_SQRT_FLOAT(x) vsqrtq_f32(x)
#define SIMD_SQRT_HALF(x) vcvt_f32_f16(vsqrt_f16(vcvt_f16_f32(x)))
#define SIMD_DIV_FLOAT(x, y) vdivq_f32(x, y) // Note: NEON does not have a direct divide intrinsic, consider a reciprocal approach
#define SIMD_DIV_HALF(x, y) vcvt_f32_f16(vdiv_f16(vcvt_f16_f32(x), vcvt_f16_f32(y)))

// Logical operations remain the same as they do not directly apply to half-precision
#define SIMD_AND(x, y) vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)))
#define SIMD_ANDNOT(x, y) vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)))
#define SIMD_OR(x, y) vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)))
#define SIMD_XOR(x, y) vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)))

#define SIMD_SET(x, h) ((h) ? SIMD_SET_HALF(x) : SIMD_SET_FLOAT(x))
#define SIMD_ADD(x, y, h) ((h) ? SIMD_ADD_HALF(x, y) : SIMD_ADD_FLOAT(x, y))
#define SIMD_MUL(x, y, h) ((h) ? SIMD_MUL_HALF(x, y) : SIMD_MUL_FLOAT(x, y))
#define SIMD_FMA(x, y, c, h) ((h) ? SIMD_FMA_HALF(x, y, c) : SIMD_FMA_FLOAT(x, y, c))
#define SIMD_SQRT(x, h) ((h) ? SIMD_SQRT_HALF(x) : SIMD_SQRT_FLOAT(x))
#define SIMD_DIV(x, y, h) ((h) ? SIMD_DIV_HALF(x, y) : SIMD_DIV_FLOAT(x, y))

union NEON_Data {
    float32x4_t data;
    // float16x8_t data_half; // Uncomment if needed, but usually not needed for vcvt intrinsics
};

template <int span>
inline void simd_store(float* dst, NEON_Data* src, bool half_precision)
{
    size_t width = SIMD_WIDTH(half_precision);
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        SIMD_STORE2(dst + width * i, src[i].data, half_precision);
    }
}

template <int span>
inline void simd_load(NEON_Data* dst, float* src, bool half_precision)
{
    size_t width = SIMD_WIDTH(half_precision);
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_LOAD2(src + width * i, half_precision);
    }
}

template <int span>
inline void simd_fma(NEON_Data* dst, NEON_Data* src_m_l, NEON_Data src_m_r, NEON_Data* src_a, bool half_precision)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_FMA(src_m_l[i].data, src_m_r.data, src_a[i].data, half_precision);
    }
}

template <int span>
inline void simd_fma(NEON_Data* dst, NEON_Data* src_m_l, NEON_Data src_m_r, NEON_Data src_a, bool half_precision)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_FMA(src_m_l[i].data, src_m_r.data, src_a.data, half_precision);
    }
}

template <int span>
inline void simd_fma(NEON_Data* dst, NEON_Data* src_m_l, NEON_Data* src_m_r, NEON_Data* src_a, bool half_precision)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_FMA(src_m_l[i].data, src_m_r[i].data, src_a[i].data, half_precision);
    }
}

template <int span>
inline void simd_sqrt(NEON_Data* dst, NEON_Data* src, bool half_precision)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_SQRT(src[i].data, half_precision);
    }
}

template <int span>
inline void simd_add(NEON_Data* dst, NEON_Data* src_a_l, NEON_Data src_a_r, bool half_precision)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_ADD(src_a_l[i].data, src_a_r.data, half_precision);
    }
}

template <int span>
inline void simd_add(NEON_Data* dst, NEON_Data* src_a_l, NEON_Data* src_a_r, bool half_precision)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_ADD(src_a_l[i].data, src_a_r[i].data, half_precision);
    }
}

template <int span>
inline void simd_mul(NEON_Data* dst, NEON_Data* src_a_l, NEON_Data src_a_r, bool half_precision)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_MUL(src_a_l[i].data, src_a_r.data, half_precision);
    }
}

template <int span>
inline void simd_mul(NEON_Data* dst, NEON_Data* src_a_l, NEON_Data* src_a_r, bool half_precision)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_MUL(src_a_l[i].data, src_a_r[i].data, half_precision);
    }
}

template <int span>
inline void simd_div(NEON_Data* dst, NEON_Data* src_a_l, NEON_Data* src_a_r, bool half_precision)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_DIV(src_a_l[i].data, src_a_r[i].data, half_precision);
    }
}

template <int span>
inline void simd_and(NEON_Data* dst, NEON_Data* src_a_l, NEON_Data src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_AND(src_a_l[i].data, src_a_r.data);
    }
}

template <int span>
inline void simd_and(NEON_Data* dst, NEON_Data* src_a_l, NEON_Data* src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_AND(src_a_l[i].data, src_a_r[i].data);
    }
}

template <int span>
inline void simd_andnot(NEON_Data* dst, NEON_Data* src_a_l, NEON_Data src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_ANDNOT(src_a_l[i].data, src_a_r.data);
    }
}

template <int span>
inline void simd_andnot(NEON_Data* dst, NEON_Data* src_a_l, NEON_Data* src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_ANDNOT(src_a_l[i].data, src_a_r[i].data);
    }
}

template <int span>
inline void simd_or(NEON_Data* dst, NEON_Data* src_a_l, NEON_Data src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_OR(src_a_l[i].data, src_a_r.data);
    }
}

template <int span>
inline void simd_or(NEON_Data* dst, NEON_Data* src_a_l, NEON_Data* src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_OR(src_a_l[i].data, src_a_r[i].data);
    }
}

template <int span>
inline void simd_xor(NEON_Data* dst, NEON_Data* src_a_l, NEON_Data src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_XOR(src_a_l[i].data, src_a_r.data);
    }
}

template <int span>
inline void simd_xor(NEON_Data* dst, NEON_Data* src_a_l, NEON_Data* src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_XOR(src_a_l[i].data, src_a_r[i].data);
    }
}
