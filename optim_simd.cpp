// Inside the for loop of custom_optimizer_step_simd function

// Precompute constants and conditional flags outside the loop
bool update_nu_frozen = (t - 1) % 2 == 0;
AVX_Data beta2_t_4 = SIMD_SET(pow(beta2, t));
AVX_Data one_minus_beta2_4 = SIMD_SET(1.0 - beta2);
AVX_Data one_4 = SIMD_SET(1.0);
AVX_Data minus_one_4 = SIMD_SET(-1.0);
AVX_Data a_t_4 = SIMD_SET((t <= 2) ? 0 : (1 - pow(mu, ceil(t - 2, 2))) / (1 - pow(mu, ceil(t, 2))));
AVX_Data b_t_4 = SIMD_SET(t > 3 ? lr / lr_prev : 1.0);
AVX_Data c_t_4 = SIMD_SET(2 * alpha * lr * weight_decay);
AVX_Data d_t_4 = SIMD_SET(2 * alpha * lr * (1 - mu) / (1 - pow(mu, ceil(t, 2))));

for (size_t i = 0; i < ROUND_DOWN(param_size, SIMD_WIDTH * span); i += SIMD_WIDTH * span) {
    // Load data into SIMD arrays
    AVX_Data grad_4[span], param_4[span], worker_params_prev_4[span], x_t_minus_2_4[span];
    AVX_Data nu_4[span], nu_frozen_4[span], nu_frozen_prev_4[span];
    simd_load<span>(grad_4, grads + i, false);
    simd_load<span>(param_4, params + i, false);
    simd_load<span>(worker_params_prev_4, worker_params_prev + i, false);
    simd_load<span>(x_t_minus_2_4, x_t_minus_2 + i, false);
    simd_load<span>(nu_4, nu + i, false);
    simd_load<span>(nu_frozen_4, nu_frozen + i, false);
    simd_load<span>(nu_frozen_prev_4, nu_frozen_prev + i, false);

    // Conditional SIMD operations
    if (update_nu_frozen) {
        for (int j = 0; j < span; ++j) {
            nu_frozen_4[j] = SIMD_MUL(nu_4[j], SIMD_DIV(one_minus_beta2_4, SIMD_SUB(one_4, beta2_t_4)));
            nu_frozen_4[j] = SIMD_ADD(SIMD_SQRT(nu_frozen_4[j]), eps_4);
            nu_frozen_prev_4[j] = nu_frozen_4[j];
        }
    }

    // Custom optimizer computations using SIMD
    for (int j = 0; j < span; ++j) {
        AVX_Data aux1_4 = SIMD_MUL(SIMD_DIV(nu_frozen_prev_4[j], nu_frozen_4[j]), SIMD_MUL(mu_4, a_t_4));
        AVX_Data aux2_4 = SIMD_FMA(x_t_minus_2_4[j], SIMD_SUB(minus_one_4, b_t_4), c_t_4, SIMD_MUL(worker_params_prev_4[j], b_t_4));
        aux1_4 = SIMD_MUL(aux1_4, aux2_4);
        aux2_4 = SIMD_MUL(SIMD_DIV(grad_4[j], nu_frozen_4[j]), d_t_4);
        aux1_4 = SIMD_SUB(SIMD_SUB(aux1_4, aux2_4), SIMD_MUL(worker_params_prev_4[j], c_t_4));
        param_4[j] = SIMD_ADD(worker_params_prev_4[j], aux1_4);
    }

    // Store the updated parameters and states back to the original memory
    simd_store<span>(params + i, param_4, false);
    if (update_nu_frozen) {
        simd_store<span>(nu_frozen + i, nu_frozen_4, false);
        simd_store<span>(nu_frozen_prev + i, nu_frozen_prev_4, false);
    }
    simd_store<span>(x_t_minus_2 + i, worker_params_prev_4, false);
    simd_store<span>(worker_params_prev + i, param_4, false);
}

// Handle any remaining elements that don't fit into the SIMD_WIDTH * span
// (Scalar computations for remaining elements)
