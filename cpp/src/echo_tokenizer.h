#pragma once
// Echo-TTS C++ — UTF-8 Byte Tokenizer
//
// Mirrors the Python tokenizer_encode() and get_text_input_ids_and_mask().
// Text is normalized (smart quotes, colons, etc.), prepended with [S1],
// UTF-8 encoded to bytes, and BOS (0) prepended.

#include <cstdint>
#include <string>
#include <vector>

struct EchoTokenizerResult {
    std::vector<int32_t> token_ids;   // (seq_len,) — padded
    std::vector<bool>    mask;        // (seq_len,) — true where valid
    std::string          normalized_text;
    int32_t              actual_length;
};

// Normalize text: smart quotes→ASCII, colons/semicolons→commas, etc.
// Optionally prepend [S1] if no bracket/paren prefix.
std::string normalize_text(const std::string & text);

// Encode text to token IDs (UTF-8 bytes with BOS=0 prepended).
std::vector<int32_t> tokenizer_encode(
    const std::string & text,
    bool append_bos = true,
    bool normalize   = true
);

// Encode text and produce token IDs + attention mask.
// max_length: 0 = no padding (use actual token count), >0 = truncate and pad to this length.
EchoTokenizerResult get_text_input_ids_and_mask(
    const std::string & text,
    int max_length = 0,
    bool normalize = true
);
