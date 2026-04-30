// Echo-TTS C++ — PCA Transform Implementation

#include "echo_pca.h"
#include <cmath>

// PCA encode: z_pca = (z_q^T - mean) @ components^T * scale
//
// Python reference:
//   z_q = fish_ae.encode_zq(audio).float()              # (B, 1024, T)
//   z_q = (z_q.transpose(1,2) - pca_mean) @ pca_components.T  # (B, T, 80)
//   z_q = z_q * latent_scale
void pca_encode(
    const EchoPCAParams & params,
    const float * z_q,
    float * z_pca,
    int batch_size,
    int time_steps
) {
    const int D = params.full_dim;   // 1024
    const int P = params.pca_dim;    // 80
    const float * comp = params.components;  // (P, D) = (80, 1024)
    const float * mean = params.mean;        // (D,)   = (1024,)
    const float scale  = params.scale;

    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < time_steps; t++) {
            // z_q is (B, D, T): element [b, d, t] = z_q[b * D * T + d * T + t]
            // We need z_q^T[b, t, :] which is z_q[b, :, t]

            for (int p = 0; p < P; p++) {
                // z_pca[b, t, p] = sum_d (z_q[b, d, t] - mean[d]) * comp[p, d]
                float sum = 0.0f;
                for (int d = 0; d < D; d++) {
                    float val = z_q[b * D * time_steps + d * time_steps + t] - mean[d];
                    sum += val * comp[p * D + d];
                }
                z_pca[b * time_steps * P + t * P + p] = sum * scale;
            }
        }
    }
}

// PCA decode: z_q = ((z_pca / scale) @ components + mean)^T
//
// Python reference:
//   z_q = (z_pca / latent_scale) @ pca_components + pca_mean  # (B, T, 1024)
//   return fish_ae.decode_zq(z_q.transpose(1,2))              # (B, 1024, T)
void pca_decode(
    const EchoPCAParams & params,
    const float * z_pca,
    float * z_q,
    int batch_size,
    int time_steps
) {
    const int D = params.full_dim;   // 1024
    const int P = params.pca_dim;    // 80
    const float * comp = params.components;  // (P, D) = (80, 1024)
    const float * mean = params.mean;        // (D,)   = (1024,)
    const float inv_scale = 1.0f / params.scale;

    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < time_steps; t++) {
            for (int d = 0; d < D; d++) {
                // z_decoded[b, t, d] = sum_p (z_pca[b, t, p] / scale) * comp[p, d] + mean[d]
                float sum = 0.0f;
                for (int p = 0; p < P; p++) {
                    float val = z_pca[b * time_steps * P + t * P + p] * inv_scale;
                    sum += val * comp[p * D + d];
                }
                sum += mean[d];

                // Transpose: z_q (B, D, T): element [b, d, t]
                z_q[b * D * time_steps + d * time_steps + t] = sum;
            }
        }
    }
}
