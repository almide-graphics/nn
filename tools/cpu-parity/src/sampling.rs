//! Optional tiny-only sampled-ID regression; uses the immutable original sampler.
use super::*;

pub fn run(raw: &[u8], path: &str, output: &Path, vocab: usize, is_ref: bool) {
    for line in fs::read_to_string(path).unwrap().lines() {
        let cols: Vec<_> = line.split('\t').collect();
        assert_eq!(cols.len(),7);
        let name=cols[0];
        let tokens=ids(cols[1]);
        let continued=ids(cols[2]);
        let ends: Vec<usize>=ids(cols[3]).into_iter().map(|x| x as usize).collect();
        let temp: f64=cols[4].parse().unwrap();
        let top_p: f64=cols[5].parse().unwrap();
        let seed: i64=cols[6].parse().unwrap();
        assert_eq!(ends.last(),Some(&tokens.len()));
        assert!(!continued.is_empty() && tokens.len()+continued.len()<=2048);
        let mut sampled=BufWriter::new(fs::File::create(output.join(format!("{name}.sampled.ids"))).unwrap());
        if is_ref {
            for (pos,&token) in tokens.iter().enumerate() {
                if ends.contains(&(pos+1)) {
                    writeln!(sampled,"{}",original::decode_sample(raw,token,pos as i64,temp,top_p,seed)).unwrap();
                } else { assert!(original::decode_argmax(raw,token,pos as i64)>=0); }
            }
        } else {
            let mut start=0;
            for &end in &ends {
                writeln!(sampled,"{}",candidate::prefill_sample(raw,&tokens[start..end],start as i64,temp,top_p,seed)).unwrap();
                start=end;
            }
            // These rejected calls must preserve the prefix before continuation.
            for (invalid,pos,code) in [(vec![],0,-3),(vec![-1],0,-3),
                (vec![vocab as i64],0,-3),(vec![0],-1,-2),(vec![0,0],2047,-2)] {
                assert_eq!(candidate::prefill_sample(raw,&invalid,pos,temp,top_p,seed),code);
            }
        }
        let decode=if is_ref {original::decode_sample} else {candidate::decode_sample};
        for (step,&token) in continued.iter().enumerate() {
            writeln!(sampled,"{}",decode(raw,token,(tokens.len()+step) as i64,temp,top_p,seed)).unwrap();
        }
        sampled.flush().unwrap();
        // Replaying the last forced token at the same absolute position preserves its
        // causal inputs. Dump all logits to check KV continuation, not just sampled IDs.
        let logits=if is_ref {original::decode_logits} else {candidate::decode_logits};
        let values=logits(raw,*continued.last().unwrap(),(tokens.len()+continued.len()-1) as i64);
        let mut dump=BufWriter::new(fs::File::create(output.join(format!("{name}.sampled.f32"))).unwrap());
        write_logits(&mut dump,&values,vocab);dump.flush().unwrap();
    }
    fs::write(output.join("sampled-complete"),"ok\n").unwrap();
}
