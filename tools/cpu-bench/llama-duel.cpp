#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>

static void log_callback(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_INFO) std::fputs(text, stderr);
}
using clock_type = std::chrono::steady_clock;
static double seconds(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}
static std::vector<llama_token> read_tokens(const std::string & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot read tokens: " + path);
    std::vector<llama_token> out;
    long long token;
    while (in >> token) {
        if (token < 0 || token > INT32_MAX) throw std::runtime_error("Invalid token ID");
        out.push_back(static_cast<llama_token>(token));
        if (out.size() > 131072) throw std::runtime_error("Too many tokens");
    }
    if (!in.eof() || out.empty()) throw std::runtime_error("Expected nonempty whitespace-separated token IDs");
    return out;
}
static int parse_int(const char * arg, int min, int max) {
    size_t end;
    int n = std::stoi(arg, &end);
    if (arg[end] != '\0' || n < min || n > max) throw std::runtime_error("Invalid integer argument");
    return n;
}
static llama_token argmax(const float * logits, int n) {
    if (!logits) throw std::runtime_error("Missing logits");
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}
static void decode(llama_context * ctx, llama_batch & batch, const llama_token * tokens, int n, int pos) {
    batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n - 1);
    }
    int rc = llama_decode(ctx, batch);
    if (rc != 0) throw std::runtime_error("llama_decode failed: " + std::to_string(rc));
    llama_synchronize(ctx);
}
static void save_ids(const std::string & path, const std::vector<llama_token> & ids) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write " + path);
    for (auto id : ids) out << id << '\n';
    if (!out) throw std::runtime_error("Failed to write " + path);
}
int main(int argc, char ** argv) {
    try {
        if (argc < 6) {
            std::cerr << "Usage: llama-duel MODEL TOKEN_FILE DECODE_STEPS THREADS OUTPUT_PREFIX [--kv f32|f16] [--ctx N] [--batch N] [--ubatch N] [--flash-attn auto|off|on] [--logits] [--warmup|--no-warmup] [--teacher-forced TOKEN_FILE]\n";
            return 2;
        }
        std::string model_path = argv[1], token_path = argv[2], prefix = argv[5];
        int steps = parse_int(argv[3], 0, 8192), threads = parse_int(argv[4], 1, 256);
        int n_ctx = 2048, n_batch = 512, n_ubatch = 512;
        bool dump_logits = false, warmup = true;
        std::string kv_type = "f32", flash = "auto", forced_path;
        for (int i = 6; i < argc; ++i) {
            std::string flag = argv[i];
            if (flag == "--logits") dump_logits = true;
            else if (flag == "--warmup") warmup = true;
            else if (flag == "--no-warmup") warmup = false;
            else {
                if (i + 1 == argc) throw std::runtime_error("Missing option value");
                const char * val = argv[++i];
                if (flag == "--kv") kv_type = val;
                else if (flag == "--ctx") n_ctx = parse_int(val, 32, 131072);
                else if (flag == "--batch") n_batch = parse_int(val, 1, 131072);
                else if (flag == "--ubatch") n_ubatch = parse_int(val, 1, 131072);
                else if (flag == "--flash-attn") flash = val;
                else if (flag == "--teacher-forced") forced_path = val;
                else throw std::runtime_error("Unknown option: " + flag);
            }
        }
        if (kv_type != "f32" && kv_type != "f16") throw std::runtime_error("KV type must be f32 or f16");
        if (flash != "auto" && flash != "off" && flash != "on") throw std::runtime_error("Flash attention must be auto, off or on");
        auto tokens = read_tokens(token_path);
        if (tokens.size() + steps > static_cast<size_t>(n_ctx)) throw std::runtime_error("Prompt plus steps exceeds context");
        std::vector<llama_token> forced;
        if (!forced_path.empty()) {
            forced = read_tokens(forced_path);
            if (forced.size() < static_cast<size_t>(steps)) throw std::runtime_error("Too few forced tokens");
        }
        llama_log_set(log_callback, nullptr);
        ggml_log_set(log_callback, nullptr);
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
        auto t0 = clock_type::now();
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(model_path.c_str(), mp), llama_model_free);
        double load_s = seconds(t0);
        if (!model) throw std::runtime_error("Model loading failed");
        int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        for (auto t : tokens) if (t >= n_vocab) throw std::runtime_error("Input token exceeds vocabulary");
        for (auto t : forced) if (t >= n_vocab) throw std::runtime_error("Forced token exceeds vocabulary");
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx;
        cp.n_batch = n_batch;
        cp.n_ubatch = n_ubatch;
        cp.n_seq_max = 1;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.type_k = cp.type_v = kv_type == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;
        cp.flash_attn_type = flash == "auto" ? LLAMA_FLASH_ATTN_TYPE_AUTO : (flash == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED);
        cp.offload_kqv = false;
        cp.op_offload = false;
        cp.no_perf = false;
        t0 = clock_type::now();
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        double context_s = seconds(t0);
        if (!ctx) throw std::runtime_error("Context creation failed");
        llama_batch batch = llama_batch_init(n_batch, 0, 1);
        if (warmup) {
            if (tokens.size() + 8 > static_cast<size_t>(n_ctx)) throw std::runtime_error("Warmup exceeds context");
            for (int pos = 0; pos < static_cast<int>(tokens.size()); pos += n_batch) {
                int n = std::min(n_batch, static_cast<int>(tokens.size()) - pos);
                decode(ctx.get(), batch, tokens.data() + pos, n, pos);
            }
            llama_token token = argmax(llama_get_logits_ith(ctx.get(), -1), n_vocab);
            for (int i = 0; i < 8; ++i) {
                decode(ctx.get(), batch, &token, 1, static_cast<int>(tokens.size()) + i);
                token = argmax(llama_get_logits_ith(ctx.get(), -1), n_vocab);
            }
            llama_memory_clear(llama_get_memory(ctx.get()), true);
        }
        std::vector<llama_token> ids;
        ids.reserve(steps + 1);
        std::vector<double> latency_ms;
        latency_ms.reserve(steps);
        std::vector<float> all_logits;
        if (dump_logits) all_logits.reserve(static_cast<size_t>(steps + 1) * n_vocab);
        auto capture = [&](const float * logits) {
            if (dump_logits) all_logits.insert(all_logits.end(), logits, logits + n_vocab);
        };
        double prefill_s, decode_s = 0.0, argmax_s = 0.0;
        auto wall_start = clock_type::now();
        t0 = clock_type::now();
        for (int pos = 0; pos < static_cast<int>(tokens.size()); pos += n_batch) {
            int n = std::min(n_batch, static_cast<int>(tokens.size()) - pos);
            decode(ctx.get(), batch, tokens.data() + pos, n, pos);
        }
        const float * logits = llama_get_logits_ith(ctx.get(), -1);
        prefill_s = seconds(t0);
        t0 = clock_type::now();
        llama_token next = argmax(logits, n_vocab);
        double first_argmax_s = seconds(t0);
        argmax_s += first_argmax_s;
        ids.push_back(next);
        capture(logits);
        auto generation_start = clock_type::now();
        for (int i = 0; i < steps; ++i) {
            llama_token input = forced.empty() ? next : forced[i];
            auto step_start = clock_type::now();
            t0 = step_start;
            decode(ctx.get(), batch, &input, 1, static_cast<int>(tokens.size()) + i);
            logits = llama_get_logits_ith(ctx.get(), -1);
            decode_s += seconds(t0);
            t0 = clock_type::now();
            next = argmax(logits, n_vocab);
            argmax_s += seconds(t0);
            latency_ms.push_back(seconds(step_start) * 1000.0);
            ids.push_back(next);
            capture(logits);
        }
        double generation_wall_s = seconds(generation_start);
        double wall_s = seconds(wall_start);
        struct rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        save_ids(prefix + ".ids.txt", ids);
        std::ofstream system_out(prefix + ".system.txt");
        system_out << llama_print_system_info() << '\n';
        if (!system_out) throw std::runtime_error("System info write failed");
        if (dump_logits) {
            std::ofstream out(prefix + ".logits.f32", std::ios::binary);
            out.write(reinterpret_cast<const char *>(all_logits.data()), static_cast<std::streamsize>(all_logits.size() * sizeof(float)));
            if (!out) throw std::runtime_error("Logits write failed");
        }
        std::ofstream out(prefix + ".json");
        out << std::setprecision(12) << "{\n"
            << "  \"engine\": \"llama.cpp\",\n"
            << "  \"source_sha\": \"8e1642198dcd4e408f8776222d6ae31b74d01187\",\n"
            << "  \"prompt_tokens\": " << tokens.size() << ",\n"
            << "  \"decode_steps\": " << steps << ",\n"
            << "  \"emitted_ids\": " << ids.size() << ",\n"
            << "  \"vocab_size\": " << n_vocab << ",\n"
            << "  \"threads\": " << threads << ",\n"
            << "  \"context_requested\": " << n_ctx << ",\n"
            << "  \"context_actual\": " << llama_n_ctx(ctx.get()) << ",\n"
            << "  \"batch\": " << n_batch << ",\n"
            << "  \"ubatch\": " << n_ubatch << ",\n"
            << "  \"kv_type\": \"" << kv_type << "\",\n"
            << "  \"flash_attention_requested\": \"" << flash << "\",\n"
            << "  \"warmup\": " << (warmup ? "true" : "false") << ",\n"
            << "  \"correctness_mode\": " << (dump_logits ? "true" : "false") << ",\n"
            << "  \"teacher_forced\": " << (!forced.empty() ? "true" : "false") << ",\n"
            << "  \"logits_rows\": " << (dump_logits ? steps + 1 : 0) << ",\n"
            << "  \"model_load_s\": " << load_s << ",\n"
            << "  \"context_init_s\": " << context_s << ",\n"
            << "  \"prefill_s\": " << prefill_s << ",\n"
            << "  \"decode_s\": " << decode_s << ",\n"
            << "  \"argmax_s\": " << argmax_s << ",\n"
            << "  \"first_argmax_s\": " << first_argmax_s << ",\n"
            << "  \"generation_wall_s\": " << generation_wall_s << ",\n"
            << "  \"inference_wall_s\": " << wall_s << ",\n"
            << "  \"prefill_tokens_per_s\": " << tokens.size() / prefill_s << ",\n"
            << "  \"decode_tokens_per_s\": " << (decode_s > 0 ? steps / decode_s : 0) << ",\n"
            << "  \"generation_with_argmax_tokens_per_s\": " << (steps > 0 ? steps / generation_wall_s : 0) << ",\n"
            << "  \"max_rss_kib\": " << usage.ru_maxrss << ",\n"
            << "  \"latency_ms\": [";
        for (size_t i = 0; i < latency_ms.size(); ++i) {
            if (i) out << ", ";
            out << latency_ms[i];
        }
        out << "]\n}\n";
        if (!out) throw std::runtime_error("Metrics write failed");
        out.close();
        std::ifstream saved(prefix + ".json");
        std::cout << saved.rdbuf();
        llama_batch_free(batch);
        ctx.reset();
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
