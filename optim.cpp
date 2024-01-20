#include <cmath>

void custom_optimizer_step(
    float* params, float* grads, 
    float* worker_params_prev, float* x_t_minus_2, 
    float* nu, float* nu_frozen, float* nu_frozen_prev, 
    int param_size, std::unordered_map<std::string, float>& state, 
    float lr, float beta2, float eps, float mu, float weight_decay, float alpha) {

    // Retrieve state values
    int t = state["step"];
    float lr_prev = state["lr_prev"];
    float lr_prev_prev = state["lr_prev_prev"];

    float a_t, b_t, c_t, d_t;
    b_t = t > 3 ? lr / lr_prev : 1.0;
    c_t = 2 * alpha * lr * weight_decay;
    d_t = 2 * alpha * lr * (1 - mu) / (1 - std::pow(mu, ceil(t, 2)));

    if (t <= 2) {
        a_t = 0;
    } else {
        a_t = (1 - std::pow(mu, ceil(t - 2, 2))) / (1 - std::pow(mu, ceil(t, 2)));
    }

    for (int i = 0; i < param_size; ++i) {
        if ((t - 1) % 2 == 0) {
            nu_frozen_prev[i] = nu_frozen[i];
        }

        float grad_square = grads[i] * grads[i];
        nu[i] = nu[i] * beta2 + grad_square;

        if ((t - 1) % 2 == 0) {
            nu_frozen[i] = nu[i] * ((1 - beta2) / (1 - std::pow(beta2, t)));
            nu_frozen[i] = std::sqrt(nu_frozen[i]) + eps;
        }

        float aux_param1 = t <= 2 ? 0 : nu_frozen_prev[i] / nu_frozen[i] * mu * a_t;
        float aux_param2 = worker_params_prev[i] * b_t - x_t_minus_2[i] * (-b_t + c_t);
        params[i] = worker_params_prev[i] + (aux_param1 * aux_param2 - grads[i] / nu_frozen[i] * d_t - worker_params_prev[i] * c_t);

        // Update x_t_minus_2 and worker_params_prev
        x_t_minus_2[i] = worker_params_prev[i];
        worker_params_prev[i] = params[i];
    }

    // Update state
    state["lr_prev_prev"] = lr_prev;
    state["lr_prev"] = lr;
    state["step"] = t + 1;
}

// Helper function for ceiling division
int ceil(int x, int y) {
    return (x + y - 1) / y;
}
