#if defined(__AVX512__) or defined(__AVX256__)
template <int span>
void Adam_Optimizer::Step_AVX(size_t* rounded_size,
                              float* _params,
                              float* grads,
                              float* worker_params_prev,
                              float* x_t_minus_2,
                              float* nu,
                              float* nu_frozen,
                              float* nu_frozen_prev,
                              size_t _param_size,
                              ds_half_precision_t* dev_params,
                              bool half_precision) {
    size_t new_rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * span);
    int rshft = half_precision ? 1 : 0;

    // Precompute SIMD constants specific to your optimizer
    AVX_Data betta2_4 = SIMD_SET(_betta2);
    AVX_Data eps_4 = SIMD_SET(_eps);
    AVX_Data lr_4 = SIMD_SET(_alpha);
    AVX_Data mu_4 = SIMD_SET(_mu);  // Assuming _mu is a parameter of your custom optimizer
    AVX_Data weight_decay_4 = SIMD_SET(_weight_decay);

    // Loop over tiles
    for (size_t t = 0; t < new_rounded_size; t += TILE) {
        size_t copy_size = ((t + TILE) > new_rounded_size) ? new_rounded_size - t : TILE;
        size_t offset = copy_size + t;

        // Synchronization for CUDA or CANN, if applicable
        // ...

#pragma omp parallel for
        for (size_t i = t; i < offset; i += SIMD_WIDTH * span) {
            // Load data into SIMD arrays
            AVX_Data grad_4[span], param_4[span], worker_params_prev_4[span], x_t_minus_2_4[span];
            AVX_Data nu_4[span], nu_frozen_4[span], nu_frozen_prev_4[span];
            simd_load<span>(grad_4, grads + (i >> rshft), half_precision);
            simd_load<span>(param_4, _params + (i >> rshft), half_precision);
            simd_load<span>(worker_params_prev_4, worker_params_prev + i, false);
            simd_load<span>(x_t_minus_2_4, x_t_minus_2 + i, false);
            simd_load<span>(nu_4, nu + i, false);
            simd_load<span>(nu_frozen_4, nu_frozen + i, false);
            simd_load<span>(nu_frozen_prev_4, nu_frozen_prev + i, false);

            // Perform SIMD operations using optimized logic
            bool update_nu_frozen = ((t - 1) % 2 == 0);
            for (int j = 0; j < span; ++j) {
                // Update nu_frozen and nu_frozen_prev
                if (update_nu_frozen) {
                    nu_frozen_prev_4[j] = nu_frozen_4[j];
                    nu_frozen_4[j] = SIMD_MUL(nu_4[j], betta2_4);
                    nu_frozen_4[j] = SIMD_ADD(nu_frozen_4[j], SIMD_MUL(SIMD_SUB(SIMD_SET(1.0), betta2_4), SIMD_MUL(grad_4[j], grad_4[j])));
                    nu_frozen_4[j] = SIMD_ADD(SIMD_SQRT(nu_frozen_4[j]), eps_4);
                }

                // Custom optimizer computations
                AVX_Data aux1_4 = SIMD_DIV(nu_frozen_prev_4[j], nu_frozen_4[j]);
                aux1_4 = SIMD_MUL(aux1_4, SIMD_MUL(mu_4, SIMD_SET((t <= 2) ? 0 : (1 - pow(_mu, ceil(t - 2, 2))) / (1 - pow(_mu, ceil(t, 2))))));
                
                AVX_Data aux2_4 = SIMD_MUL(worker_params_prev_4[j], SIMD_SET(t > 3 ? _alpha / _prev_lr : 1.0));
                aux2_4 = SIMD_FMA(x_t_minus_2_4[j], SIMD_SUB(SIMD_SET(-1.0), SIMD_SET(t > 3 ? _alpha / _prev_lr : 1.0)), SIMD_SET(2 * _alpha * _weight_decay), aux2_4);
                aux1_4 = SIMD_MUL(aux1_4, aux2_4);

                aux2_4 = SIMD_DIV(grad_4[j], nu_frozen_4[j]);
                aux1_4 = SIMD_SUB(aux1_4, SIMD_MUL(aux2_4, SIMD_SET(2 * _alpha * (1 - _mu) / (1 - pow(_mu, ceil(t, 2))))));
                aux1_4 = SIMD_SUB(aux1_4, SIMD_MUL(worker_params_prev_4[j], SIMD_SET(2 * _alpha * _weight_decay)));

                param_4[j] = SIMD_ADD(worker_params_prev_4[j], aux1_4);
            }

            // Store the updated parameters and states back to the original memory
            simd_store<span>(_params + (i >> rshft), param_4, half_precision);
            simd_store<span>(worker_params_prev + i, worker_params_prev_4, false);
            simd_store<span>(x_t_minus_2 + i, x_t_minus_2_4, false);
            simd_store<span>(nu + i, nu_4, false);
            simd_store<span>(nu_frozen + i, nu_frozen_4, false);
            simd_store<span>(nu_frozen_prev + i, nu_frozen_prev_4, false);

#if defined(__ENABLE_CUDA__) or defined(__ENABLE_CANN__)
            if (dev_params) {
                simd_store<span>(_doubled_buffer[_buf_index] + (i - t), param_4, half_precision);
            }
#endif
        }

        // Post-loop operations for CUDA or CANN
        // ...
    }
    *rounded_size = new_rounded_size;
}
#endif
