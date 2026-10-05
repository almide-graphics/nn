//! Isolated original/candidate decoder replay comparison. No inference I/O is timed.
#![allow(dead_code, unused_mut)]
use std::{env, fs, io::{BufWriter, Write}, path::Path, time::{Instant, SystemTime, UNIX_EPOCH}};

// Exact conversion from nn/native/gpu.rs and the existing parity driver.
mod gpu {
    pub fn fp16_to_f32_pub(raw: u16) -> f32 {
        let sign = (raw >> 15) as u32;
        let exp = ((raw >> 10) & 0x1F) as u32;
        let man = (raw & 0x3FF) as u32;
        let bits = if exp == 0 {
            if man == 0 { sign << 31 } else {
                let mut m = man;
                let mut e = 127 - 15 + 1;
                while m & 0x400 == 0 { m <<= 1; e -= 1; }
                (sign << 31) | ((e as u32) << 23) | ((m & 0x3FF) << 13)
            }
        } else if exp == 31 {
            (sign << 31) | (0xFF << 23) | (man << 13)
        } else {
            (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13)
        };
        f32::from_bits(bits)
    }
}
mod original {
    include!(env!("CPU_DECODE_ORIGINAL"));
    // Read-only harness helper. The included immutable source is unchanged.
    pub fn snapshot_logit_bits() -> Vec<u32> {
        let guard = cell().lock().unwrap();
        guard.as_ref().unwrap().logits.iter().map(|v| v.to_bits()).collect()
    }
}
mod candidate {
    include!(env!("CPU_DECODE_CANDIDATE"));
    // Read-only harness helper. Never called in a timed region.
    pub fn snapshot_logit_bits() -> Vec<u32> {
        let guard = cell().lock().unwrap();
        guard.as_ref().unwrap().logits.iter().map(|v| v.to_bits()).collect()
    }
}

const THREADS: usize = 4;
const WARM: usize = 8;
const STEPS: usize = 128;
const BLOCKS: usize = 5;
type Decode = fn(&[u8], i64, i64) -> i64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Engine { Original, Candidate }
impl Engine {
    fn name(self) -> &'static str {
        match self { Self::Original => "original", Self::Candidate => "candidate" }
    }
    fn decode(self) -> Decode {
        match self { Self::Original => original::decode_argmax, Self::Candidate => candidate::decode_argmax }
    }
    fn logit_bits(self) -> Vec<u32> {
        match self { Self::Original => original::snapshot_logit_bits(), Self::Candidate => candidate::snapshot_logit_bits() }
    }
}

struct Config { values: Vec<String>, gammas: Vec<i64>, weights: Vec<i64> }
impl Config {
    fn read(path: &str) -> Self {
        let text = fs::read_to_string(path).unwrap();
        let lines: Vec<_> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        let values: Vec<_> = lines[0].split_whitespace().map(str::to_owned).collect();
        assert_eq!(values.len(), 11);
        Self { values, gammas: ids(lines[1]), weights: ids(lines[2]) }
    }
    fn vocab(&self) -> i64 { self.values[6].parse().unwrap() }
    fn load(&self, engine: Engine, raw: &[u8]) {
        let c: Vec<i64> = self.values[..7].iter().map(|x| x.parse().unwrap()).collect();
        let load = match engine { Engine::Original => original::load_model, Engine::Candidate => candidate::load_model };
        assert_eq!(load(raw, c[0], c[1], c[2], c[3], c[4], c[5], c[6],
            self.values[7].parse().unwrap(), self.values[8].parse().unwrap(),
            self.values[9].parse().unwrap(), self.values[10].parse().unwrap(),
            &self.gammas, &self.weights), 1, "{} load failed", engine.name());
    }
}

fn ids(text: &str) -> Vec<i64> { text.split_whitespace().map(|x| x.parse().unwrap()).collect() }
fn case_tokens(path: &str, name: &str, vocab: i64) -> Vec<i64> {
    let text = fs::read_to_string(path).unwrap();
    let matching: Vec<_> = text.lines().filter(|s| s.split('\t').next() == Some(name)).collect();
    assert_eq!(matching.len(), 1, "missing/duplicate case {name}");
    let cols: Vec<_> = matching[0].split('\t').collect();
    assert_eq!(cols.len(), 4);
    let tokens = ids(cols[1]);
    assert!(!tokens.is_empty() && tokens.len() + STEPS <= 2048);
    assert!(tokens.iter().all(|&v| v >= 0 && v < vocab));
    tokens
}
fn prompt(decode: Decode, raw: &[u8], tokens: &[i64], vocab: i64) -> i64 {
    let mut predicted = -1;
    for (pos, &token) in tokens.iter().enumerate() {
        predicted = decode(raw, token, pos as i64);
        assert!(predicted >= 0 && predicted < vocab);
    }
    predicted
}

// Always restart at the same saved prediction/absolute position. All output storage is
// caller-allocated. No EOS termination, logging, validation, allocation by this driver,
// clocks, or file I/O occurs inside the loop. Allocations in the decoder API itself
// remain included. The 128 returned IDs are the outputs of exactly 128 decode calls;
// the untimed initial prediction from the last prompt token is recorded separately.
#[inline(never)]
fn replay(decode: Decode, raw: &[u8], initial: i64, start_pos: usize, output: &mut [i64]) {
    let mut token = initial;
    for (step, next) in output.iter_mut().enumerate() {
        token = decode(raw, token, (start_pos + step) as i64);
        *next = token;
    }
    std::hint::black_box(token);
}
fn assert_ids(actual: &[i64], expected: &[i64], label: &str) {
    assert_eq!(actual.len(), expected.len());
    for (step, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a, e, "{label}, step={step}");
    }
}
fn assert_logit_bits(actual: &[u32], expected: &[u32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: logit count");
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(f32::from_bits(a).is_finite() && f32::from_bits(e).is_finite(), "{label}: non-finite logit {index}");
        assert_eq!(a, e, "{label}: logit bits at {index}");
    }
}
fn write_ids(writer: &mut impl Write, output: &[i64]) {
    for (i, token) in output.iter().enumerate() {
        if i != 0 { write!(writer, " ").unwrap(); }
        write!(writer, "{token}").unwrap();
    }
}

fn verify_replay(raw: &[u8], cfg: &Config, cases: &str, output: &Path) {
    let mut log = BufWriter::new(fs::File::create(output.join("replay-validation.tsv")).unwrap());
    writeln!(log, "case\tengine\tmode\tinitial_id\tfinal_logit_count\tfinal_logit_bits_identical\tids").unwrap();
    for name in ["input-128", "input-1024"] {
        let tokens = case_tokens(cases, name, cfg.vocab());
        let mut reference_initial = None;
        let mut reference = [0i64; STEPS];
        let mut reference_bits = Vec::new();
        for engine in [Engine::Original, Engine::Candidate] {
            cfg.load(engine, raw);
            let decode = engine.decode();
            let initial = prompt(decode, raw, &tokens, cfg.vocab());
            let mut expected = [0i64; STEPS];
            replay(decode, raw, initial, tokens.len(), &mut expected);
            let expected_bits = engine.logit_bits();
            assert_eq!(expected_bits.len(), cfg.vocab() as usize);
            if engine == Engine::Original { reference_initial = Some(initial); reference = expected; reference_bits = expected_bits.clone(); }
            assert_eq!(Some(initial), reference_initial, "fresh initial {name}");
            assert_ids(&expected, &reference, &format!("fresh {name} {}", engine.name()));
            assert_logit_bits(&expected_bits, &reference_bits, "fresh tiny cross-engine logits");
            writeln!(log, "{name}\t{}\tfresh\t{initial}\t{}\ttrue\t{}", engine.name(), expected_bits.len(), expected.iter().map(i64::to_string).collect::<Vec<_>>().join(" ")).unwrap();

            // Deliberately contaminate every generated slot with a different teacher-forced
            // sequence, leaving prompt KV untouched. This tests ignoring stale future KV.
            for step in 0..STEPS {
                let poison = (initial + 17 + step as i64 * 31) % cfg.vocab();
                assert!(decode(raw, poison, (tokens.len() + step) as i64) >= 0);
            }
            for attempt in 0..3 {
                let mut warm = [0i64; WARM];
                replay(decode, raw, initial, tokens.len(), &mut warm);
                assert_ids(&warm, &expected[..WARM], "validation warmup");
                let mut got = [0i64; STEPS];
                replay(decode, raw, initial, tokens.len(), &mut got);
                assert_ids(&got, &expected, &format!("replay {name} {} {attempt}", engine.name()));
                assert_logit_bits(&engine.logit_bits(), &expected_bits, "tiny replay logits");
                write!(log, "{name}\t{}\treplay-{attempt}\t{initial}\t{}\ttrue\t", engine.name(), expected_bits.len()).unwrap();
                write_ids(&mut log, &got); writeln!(log).unwrap();
            }

            // Independent fresh model state + full prompt is the final comparison, not
            // merely a second replay that could share the same latent contamination.
            cfg.load(engine, raw);
            let fresh_initial = prompt(decode, raw, &tokens, cfg.vocab());
            assert_eq!(fresh_initial, initial);
            let mut fresh = [0i64; STEPS];
            replay(decode, raw, fresh_initial, tokens.len(), &mut fresh);
            assert_ids(&fresh, &expected, "fresh reload comparison");
            assert_logit_bits(&engine.logit_bits(), &expected_bits, "tiny fresh reload logits");
            write!(log, "{name}\t{}\tfresh-reload\t{initial}\t{}\ttrue\t", engine.name(), expected_bits.len()).unwrap();
            write_ids(&mut log, &fresh); writeln!(log).unwrap();
            log.flush().unwrap();
            eprintln!("replay validation passed: {name} {}", engine.name());
        }
    }
}

fn epoch_ns() -> u128 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() }
fn benchmark(raw: &[u8], cfg: &Config, cases: &str, output: &Path) {
    cfg.load(Engine::Original, raw);
    cfg.load(Engine::Candidate, raw);
    let mut runs = BufWriter::new(fs::File::create(output.join("runs.tsv")).unwrap());
    let mut reference_log = BufWriter::new(fs::File::create(output.join("reference-ids.tsv")).unwrap());
    writeln!(runs, "case\tprompt_tokens\tblock\torder\tslot\tengine\tengine_run\tstart_epoch_ns\tstart_monotonic_ns\tduration_ns\tdecode_calls\twarmup_calls\tinitial_id\tfinal_logit_count\tfinal_logit_bits_identical\tids").unwrap();
    writeln!(reference_log, "case\tinitial_id\tprompt_ids\tdecode_output_ids").unwrap();
    let origin = Instant::now();
    for (case_index, name) in ["input-128", "input-1024"].iter().enumerate() {
        let tokens = case_tokens(cases, name, cfg.vocab());
        let initial_original = prompt(original::decode_argmax, raw, &tokens, cfg.vocab());
        let initial_candidate = prompt(candidate::decode_argmax, raw, &tokens, cfg.vocab());
        assert_eq!(initial_original, initial_candidate, "initial prediction {name}");
        let initial = initial_original;
        let mut expected = [0i64; STEPS];
        replay(original::decode_argmax, raw, initial, tokens.len(), &mut expected);
        let expected_bits = original::snapshot_logit_bits();
        assert_eq!(expected_bits.len(), cfg.vocab() as usize);
        assert!(expected.iter().all(|&v| v >= 0 && v < cfg.vocab()));
        let mut candidate_initial = [0i64; STEPS];
        replay(candidate::decode_argmax, raw, initial, tokens.len(), &mut candidate_initial);
        assert_ids(&candidate_initial, &expected, "untimed real-model decode reference");
        assert_logit_bits(&candidate::snapshot_logit_bits(), &expected_bits, "untimed real-model logits");
        write!(reference_log, "{name}\t{initial}\t").unwrap();
        write_ids(&mut reference_log, &tokens); write!(reference_log, "\t").unwrap();
        write_ids(&mut reference_log, &expected); writeln!(reference_log).unwrap();
        reference_log.flush().unwrap();
        eprintln!("prompt/reference ready: {name}; {} prompt tokens; 128 matching decode outputs", tokens.len());

        let mut count = [0usize; 2];
        for block in 0..BLOCKS {
            let abba = (block + case_index) % 2 == 0;
            let order = if abba { "ABBA" } else { "BAAB" };
            let engines = if abba {
                [Engine::Original, Engine::Candidate, Engine::Candidate, Engine::Original]
            } else {
                [Engine::Candidate, Engine::Original, Engine::Original, Engine::Candidate]
            };
            for (slot, engine) in engines.iter().copied().enumerate() {
                let decode = engine.decode();
                let mut warm = [0i64; WARM];
                let mut got = [0i64; STEPS];
                replay(decode, raw, initial, tokens.len(), &mut warm);
                assert_ids(&warm, &expected[..WARM], "real warmup replay");
                let epoch = epoch_ns();
                let monotonic = origin.elapsed().as_nanos();
                // The only measured region is this replay of exactly 128 decode calls.
                let start = Instant::now();
                replay(decode, raw, initial, tokens.len(), &mut got);
                let duration = start.elapsed();
                assert_ids(&got, &expected, "timed replay IDs");
                // Snapshot, allocation and bitwise full-vocabulary check are AFTER timing.
                assert_logit_bits(&engine.logit_bits(), &expected_bits, "timed replay final logits");
                let index = if engine == Engine::Original { 0 } else { 1 };
                count[index] += 1;
                write!(runs, "{name}\t{}\t{block}\t{order}\t{slot}\t{}\t{}\t{epoch}\t{monotonic}\t{}\t{STEPS}\t{WARM}\t{initial}\t{}\ttrue\t", tokens.len(), engine.name(), count[index], duration.as_nanos(), expected_bits.len()).unwrap();
                write_ids(&mut runs, &got); writeln!(runs).unwrap(); runs.flush().unwrap();
                eprintln!("{name} block={block} {order} slot={slot} {} run={} seconds={:.6} tok/s={:.3}", engine.name(), count[index], duration.as_secs_f64(), STEPS as f64 / duration.as_secs_f64());
            }
        }
        assert_eq!(count, [10, 10]);
    }
}

fn main() {
    let a: Vec<String> = env::args().collect();
    assert_eq!(a.len(), 9, "usage: driver tiny_model tiny_config tiny_cases real_model real_config real_cases output_dir validate-only|benchmark");
    assert!(a[8] == "validate-only" || a[8] == "benchmark");
    assert_eq!(env::var("ALMIDE_LOCKSTEP_THREADS").unwrap(), THREADS.to_string());
    assert_eq!(env::var("RAYON_NUM_THREADS").unwrap(), THREADS.to_string());
    assert!(std::thread::available_parallelism().unwrap().get() >= THREADS);
    let status = fs::read_to_string("/proc/self/status").unwrap();
    let affinity = status.lines().find(|l| l.starts_with("Cpus_allowed_list:")).unwrap();
    assert_eq!(affinity.split_whitespace().nth(1).unwrap(), env::var("CPU_DECODE_EXPECTED_AFFINITY").unwrap(), "driver must inherit the requested four-CPU affinity");
    rayon::ThreadPoolBuilder::new().num_threads(THREADS).build_global().unwrap();
    assert_eq!(rayon::current_num_threads(), THREADS);
    let output = Path::new(&a[7]); fs::create_dir_all(output).unwrap();
    fs::write(output.join("process-status.txt"), status).unwrap();
    {
        let raw = fs::read(&a[1]).unwrap();
        verify_replay(&raw, &Config::read(&a[2]), &a[3], output);
    }
    fs::write(output.join("replay-validation.complete"), "ok\n").unwrap();
    if a[8] == "benchmark" {
        // One allocation/read of the real model, shared read-only by both native modules.
        let raw = fs::read(&a[4]).unwrap();
        benchmark(&raw, &Config::read(&a[5]), &a[6], output);
        fs::write(output.join("complete"), "ok\n").unwrap();
    }
}
