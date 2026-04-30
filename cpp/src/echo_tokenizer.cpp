// Echo-TTS C++ — UTF-8 Byte Tokenizer Implementation

#include "echo_tokenizer.h"
#include <algorithm>

// ── Helpers ─────────────────────────────────────────────────────────

// Replace all occurrences of `from` with `to` in `str` (in-place).
static void replace_all(std::string & str, const std::string & from, const std::string & to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = str.find(from, pos)) != std::string::npos) {
        str.replace(pos, from.size(), to);
        pos += to.size();
    }
}

// Check if string starts with a given prefix.
static bool starts_with(const std::string & str, const std::string & prefix) {
    return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}

// ── Public API ──────────────────────────────────────────────────────

std::string normalize_text(const std::string & text) {
    std::string t = text;

    // Smart quotes and special chars → ASCII
    // UTF-8 sequences for: … ' " " — 
    replace_all(t, "\xe2\x80\xa6", "...");       // …
    replace_all(t, "\xe2\x80\x99", "'");          // '
    replace_all(t, "\xe2\x80\x9c", "\"");         // "
    replace_all(t, "\xe2\x80\x9d", "\"");         // "
    replace_all(t, "\xe2\x80\x94", ", ");         // —

    // Punctuation normalization
    replace_all(t, "\n", " ");
    replace_all(t, ":", ",");
    replace_all(t, ";", ",");

    // Prepend [S1] if no bracket/parenthesis prefix and no S1/S2 tag
    if (!starts_with(t, "[") && !starts_with(t, "(") &&
        t.find("S1") == std::string::npos &&
        t.find("S2") == std::string::npos) {
        t = "[S1] " + t;
    }

    return t;
}

std::vector<int32_t> tokenizer_encode(
    const std::string & text,
    bool append_bos,
    bool normalize
) {
    std::string t = normalize ? normalize_text(text) : text;

    // UTF-8 encode: each byte becomes a token
    std::vector<int32_t> tokens;
    tokens.reserve(t.size() + 1);

    if (append_bos) {
        tokens.push_back(0);  // BOS token
    }

    for (unsigned char c : t) {
        tokens.push_back(static_cast<int32_t>(c));
    }

    return tokens;
}

EchoTokenizerResult get_text_input_ids_and_mask(
    const std::string & text,
    int max_length,
    bool normalize
) {
    EchoTokenizerResult result;

    // Normalize first so we can capture the normalized text
    result.normalized_text = normalize ? normalize_text(text) : text;

    // Encode (skip double-normalization since we already normalized)
    std::vector<int32_t> encoded = tokenizer_encode(result.normalized_text, true, false);

    int encoded_len = static_cast<int>(encoded.size());

    if (max_length > 0) {
        // Padded mode: truncate to max_length and pad with zeros
        int actual_len = std::min(encoded_len, max_length);
        result.actual_length = actual_len;

        result.token_ids.resize(max_length, 0);
        result.mask.resize(max_length, false);

        for (int i = 0; i < actual_len; i++) {
            result.token_ids[i] = encoded[i];
            result.mask[i] = true;
        }
    } else {
        // Unpadded mode (max_length <= 0): return exactly the encoded tokens
        result.actual_length = encoded_len;

        result.token_ids = std::move(encoded);
        result.mask.resize(encoded_len, true);
    }

    return result;
}
