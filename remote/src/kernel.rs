//! 行列積(f32、行優先)。手元もスマホ側(worker)も同じ実装を使い、内部は open-cpu の `axpy_f32`
//! (AVX-512/AVX2+FMA/NEON を実行時に自動選択)で行う。

use std::sync::Mutex;

/// 一度に取り出す出力行数(動的な仕事の取り合い。big.LITTLE のように速さの違うコアでも偏らない)。
const BLOCK_ROWS: usize = 4;

/// `c[m×n] = a[m×k] × b[k×n]`(スレッド数 `threads`。0 なら利用可能な全コア)。
pub fn gemm_local(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, threads: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "a の大きさ");
    assert_eq!(b.len(), k * n, "b の大きさ");
    let mut c = vec![0.0f32; m * n];
    if m == 0 || n == 0 {
        return c;
    }
    let threads = if threads == 0 { std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1) } else { threads }.min(m).max(1);
    let queue = Mutex::new(c.chunks_mut(BLOCK_ROWS * n).enumerate());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let item = queue.lock().map(|mut q| q.next()).unwrap_or(None);
                let Some((bi, chunk)) = item else { break };
                for (ri, row) in chunk.chunks_mut(n).enumerate() {
                    let i = bi * BLOCK_ROWS + ri;
                    let arow = &a[i * k..(i + 1) * k];
                    for (p, &av) in arow.iter().enumerate() {
                        if av != 0.0 {
                            open_cpu::axpy_f32(row, &b[p * n..(p + 1) * n], av);
                        }
                    }
                }
            });
        }
    });
    c
}

/// 参照実装(3重ループ)。試験用。
pub fn gemm_naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += av * b[p * n + j];
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(len: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761) | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                ((s % 2001) as f32 - 1000.0) / 250.0
            })
            .collect()
    }

    #[test]
    fn matches_naive_for_odd_shapes() {
        for &(m, k, n) in &[(1, 1, 1), (3, 5, 7), (9, 16, 33), (17, 31, 65), (64, 64, 64), (5, 129, 3)] {
            let (a, b) = (fill(m * k, 7), fill(k * n, 9));
            let want = gemm_naive(&a, &b, m, k, n);
            for threads in [1, 3, 0] {
                let got = gemm_local(&a, &b, m, k, n, threads);
                for (x, y) in got.iter().zip(&want) {
                    assert!((x - y).abs() <= 1e-2 * (1.0 + y.abs()), "{m}x{k}x{n} threads={threads}: {x} vs {y}");
                }
            }
        }
    }

    #[test]
    fn empty_is_empty() {
        assert!(gemm_local(&[], &[], 0, 0, 0, 0).is_empty());
    }
}
