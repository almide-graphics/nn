//! Correctness-only driver. No reported inference timings; all dumps are binary f32 LE.
#![allow(dead_code, unused_mut)]
use std::{env, fs, io::{BufWriter, Write}, path::Path};

// Exact conversion used by native/gpu.rs; avoids pulling a GPU runtime into this CPU test.
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
mod original { include!(env!("CPU_PARITY_ORIGINAL")); }
mod candidate { include!(env!("CPU_PARITY_CANDIDATE")); }

fn ids(s: &str) -> Vec<i64> {
    s.split_whitespace().map(|x| x.parse().unwrap()).collect()
}
fn write_logits(out: &mut BufWriter<fs::File>, logits: &[f64], vocab: usize) {
    assert_eq!(logits.len(), vocab, "wrong or missing logits");
    for &v in logits { out.write_all(&(v as f32).to_le_bytes()).unwrap(); }
}
fn main() {
    let a: Vec<String> = env::args().collect();
    assert_eq!(a.len(), 6, "mode model config cases out_dir");
    let threads: usize = env::var("ALMIDE_LOCKSTEP_THREADS").expect("explicit thread count required").parse().unwrap();
    let available = std::thread::available_parallelism().expect("cannot determine safe worker limit").get();
    assert!(threads > 0 && threads <= available,
        "requested {threads} threads exceeds safe available_parallelism={available}; use --threads 1 on small runners");
    let rayon_threads: usize = env::var("RAYON_NUM_THREADS").expect("explicit Rayon size required").parse().unwrap();
    assert_eq!(rayon_threads,threads,"Rayon and lockstep pool sizes must match");
    let is_ref = match a[1].as_str() {
        "reference" => true, "candidate" => false, _ => panic!("unknown mode")
    };
    let raw = fs::read(&a[2]).unwrap();
    let config = fs::read_to_string(&a[3]).unwrap();
    let lines: Vec<_> = config.lines().collect();
    let cfg: Vec<_> = lines[0].split_whitespace().collect();
    let c: Vec<i64> = cfg[..7].iter().map(|x| x.parse().unwrap()).collect();
    let gammas = ids(lines[1]);
    let weights = ids(lines[2]);
    let load = if is_ref { original::load_model } else { candidate::load_model };
    if !is_ref {
        assert_eq!(candidate::prefill_argmax(&raw,&[0],0),-1);
        assert!(candidate::prefill_logits(&raw,&[0],0).is_empty());
    }
    assert_eq!(load(&raw, c[0],c[1],c[2],c[3],c[4],c[5],c[6],
        cfg[7].parse().unwrap(),cfg[8].parse().unwrap(),cfg[9].parse().unwrap(),
        cfg[10].parse().unwrap(), &gammas,&weights), 1);
    let vocab = c[6] as usize;
    let cases = fs::read_to_string(&a[4]).unwrap();
    let output = Path::new(&a[5]);
    fs::create_dir_all(output).unwrap();
    let mut log = BufWriter::new(fs::File::create(output.join("completed.tsv")).unwrap());
    for line in cases.lines() {
        let cols: Vec<_> = line.split('\t').collect();
        assert_eq!(cols.len(), 4);
        let name = cols[0];
        let tokens = ids(cols[1]);
        let continued = ids(cols[2]);
        let chunk_ends: Vec<usize> = ids(cols[3]).into_iter().map(|i| i as usize).collect();
        assert!(!tokens.is_empty());
        assert!(tokens.len()+continued.len() <= 2048);
        assert_eq!(*chunk_ends.last().unwrap(),tokens.len());
        assert!(tokens.iter().chain(continued.iter()).all(|&t| t >= 0 && t < vocab as i64));
        let mut dump = BufWriter::new(fs::File::create(output.join(format!("{name}.f32"))).unwrap());
        if is_ref {
            for (pos, &token) in tokens.iter().enumerate() {
                if chunk_ends.contains(&(pos+1)) {
                    let logits = original::decode_logits(&raw,token,pos as i64);
                    write_logits(&mut dump,&logits,vocab);
                } else { assert!(original::decode_argmax(&raw,token,pos as i64)>=0); }
            }
        } else {
            let mut start = 0;
            for &end in &chunk_ends {
                let logits = candidate::prefill_logits(&raw,&tokens[start..end],start as i64);
                write_logits(&mut dump,&logits,vocab);
                start = end;
            }
        }
        if !is_ref {
            // Invalid calls must reject before mutating the valid prefix's KV state.
            for (invalid, pos, code) in [
                (vec![], 0, -3), (vec![-1], 0, -3), (vec![vocab as i64], 0, -3),
                (vec![0], -1, -2), (vec![0], i64::MAX, -2), (vec![0,0], 2047, -2),
            ] {
                assert!(candidate::prefill_logits(&raw,&invalid,pos).is_empty());
                assert_eq!(candidate::prefill_argmax(&raw,&invalid,pos),code);
            }
        }
        let decode = if is_ref { original::decode_logits } else { candidate::decode_logits };
        for (step,&token) in continued.iter().enumerate() {
            let logits=decode(&raw,token,(tokens.len()+step) as i64);
            write_logits(&mut dump,&logits,vocab);
        }
        dump.flush().unwrap();
        if !is_ref {
            // Exercise the production argmax entry point and its continued KV state independently.
            let mut top = BufWriter::new(fs::File::create(output.join(format!("{name}.argmax"))).unwrap());
            let mut start = 0;
            for &end in &chunk_ends {
                writeln!(top,"{}",candidate::prefill_argmax(&raw,&tokens[start..end],start as i64)).unwrap();
                start = end;
            }
            for (step,&token) in continued.iter().enumerate() {
                writeln!(top,"{}",candidate::decode_argmax(&raw,token,(tokens.len()+step) as i64)).unwrap();
            }
            top.flush().unwrap();
        }
        writeln!(log,"{name}\t{}\t{}",tokens.len(),chunk_ends.len()+continued.len()).unwrap();
        log.flush().unwrap();
        eprintln!("{} {name}: {} prompt tokens, {} checkpoints",a[1],tokens.len(),chunk_ends.len()+continued.len());
    }
    // Public APIs must reject a token beyond the fixed context, without returning logits.
    let decode = if is_ref { original::decode_logits } else { candidate::decode_logits };
    assert!(decode(&raw,0,2048).is_empty());
    if !is_ref {
        assert!(candidate::prefill_logits(&raw,&[0],2048).is_empty());
        assert_eq!(candidate::prefill_argmax(&raw,&[0],2048),-2);
    }
    fs::write(output.join("complete"),"ok\n").unwrap();
}
