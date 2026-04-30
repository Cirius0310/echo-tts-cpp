# Echo TTS — Example Audio Samples

Generated with Echo TTS C++ using the default 40-step Euler sampler with CFG (text=3.0, speaker=8.0).

## Voices

| Voice | Reference WAV | Description |
|-------|---------------|-------------|
| `echo` | `ana_eleven.wav` | Warm, calm narrator voice (~59s reference) |
| `herta` | `herta.wav` | Playful, expressive voice (~26s reference) |
| `kore` | `kore_gemini.wav` | Deep, wise voice (~20s reference) |

## Samples

### 01 — Echo: Welcome (`01_echo_welcome.wav`)
> "Welcome to Echo TTS, an open-source text-to-speech synthesis engine powered by a diffusion transformer architecture. It generates high-quality speech in real-time on consumer hardware."
- **Voice:** echo
- **Duration:** 13.6s
- **Generation time:** 7.2s (0.53x real-time)
- **Style:** Neutral narration, product introduction

### 02 — Herta: Dialogue (`02_herta_dialogue.wav`)
> "Oh my, you're still here? How delightful. I was starting to think you'd run off to bother some other Aeon. Come, sit down. Let's review those stellar charts together."
- **Voice:** herta
- **Duration:** 12.8s
- **Generation time:** 5.9s (0.46x real-time)
- **Style:** Casual dialogue, warm invitation

### 03 — Echo: Technical Explanation (`03_echo_technical.wav`)
> "The diffusion transformer generates speech by iteratively denoising a random latent vector over forty Euler steps. Each step uses classifier-free guidance with both text and speaker conditioning, producing natural-sounding audio at forty-four kilohertz sample rate."
- **Voice:** echo
- **Duration:** 20.0s
- **Generation time:** 7.2s (0.36x real-time)
- **Style:** Technical explanation, clear enunciation

### 04 — Herta: Playful (`04_herta_playful.wav`)
> "Oh, you want me to generate speech for you? Of course you do! Just type your text, pick a voice, and I'll make the magic happen. It's almost as fun as collecting stellar data. Almost."
- **Voice:** herta
- **Duration:** 13.7s
- **Generation time:** 5.8s (0.42x real-time)
- **Style:** Playful, fourth-wall breaking, meta

### 05 — Echo: Architecture Overview (`05_echo_architecture.wav`)
> "Echo TTS is built from the ground up in C plus plus, using GGML for efficient tensor operations and a custom GGUF model loader. The architecture combines a fourteen-layer bidirectional text encoder, a fourteen-layer causal speaker encoder, and a twenty-four-layer diffusion transformer decoder with joint attention and low-rank adaptive layer normalization. All inference runs locally on GPU, with CUDA acceleration and ONNX runtime for the DAC audio codec."
- **Voice:** echo
- **Duration:** 29.7s (max model capacity, 640 latent frames)
- **Generation time:** 6.9s (0.23x real-time)
- **Style:** Detailed technical description

### 06 — Kore: Wisdom (`06_kore_wisdom.wav`)
> "Greetings, traveler. The stars align favorably for your query today. What knowledge do you seek from the archives? I have records spanning a thousand cycles of galactic history at my disposal."
- **Voice:** kore
- **Duration:** 15.4s
- **Generation time:** 6.0s (0.39x real-time)
- **Style:** Formal greeting, wisdom-keeper persona

## Speed Notes

All samples were generated on an NVIDIA RTX 4080 SUPER (16 GB VRAM). Generation runs at 0.23–0.53x real-time, meaning 1 second of audio takes 0.23–0.53 seconds to generate.

For text longer than ~400 characters, the server mode automatically splits into chunks with proper sentence boundaries to avoid exceeding the model's 30-second generation limit.
