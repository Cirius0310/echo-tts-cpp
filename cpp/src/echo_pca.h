#pragma once
// Echo-TTS C++ — PCA Transform
//
// PCA encode: z_pca = (z_q^T - mean) @ components^T * scale
// PCA decode: z_q = ((z_pca / scale) @ components + mean)^T
//
// These operate on raw float arrays on CPU. The PCA state (components, mean,
// scale) is loaded from the GGUF file by EchoModel.

#include <vector>

struct EchoPCAParams {
    const float * components;   // (80, 1024) row-major
    const float * mean;         // (1024,)
    float         scale;
    int           pca_dim;      // 80
    int           full_dim;     // 1024
};

// Encode: z_q (batch, full_dim, time) → z_pca (batch, time, pca_dim)
// z_q is in (batch, channels=1024, time) layout (from DAC encoder)
// z_pca is in (batch, time, pca_dim=80) layout (for EchoDiT)
void pca_encode(
    const EchoPCAParams & params,
    const float * z_q,        // (batch, 1024, time)
    float * z_pca,            // (batch, time, 80)
    int batch_size,
    int time_steps
);

// Decode: z_pca (batch, time, pca_dim) → z_q (batch, full_dim, time)
void pca_decode(
    const EchoPCAParams & params,
    const float * z_pca,      // (batch, time, 80)
    float * z_q,              // (batch, 1024, time)
    int batch_size,
    int time_steps
);
