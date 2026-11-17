use std::time::Instant;

fn main() {
    println!("byte impl: {}", open_cpu::selected_byte_impl());
    let mut d = vec![b'a'; 64 * 1024 * 1024];
    let n = d.len();
    d[n - 1] = b'&';
    for (name, f) in [
        ("find  open-cpu", (|d: &[u8]| open_cpu::find_byte(d, b'&').unwrap_or(0)) as fn(&[u8]) -> usize),
        ("find  scalar  ", |d: &[u8]| open_cpu::find_byte_scalar(d, b'&').unwrap_or(0)),
        ("count open-cpu", |d: &[u8]| open_cpu::count_byte(d, b'a')),
        ("count scalar  ", |d: &[u8]| open_cpu::count_byte_scalar(d, b'a')),
    ] {
        let t = Instant::now();
        let mut r = 0;
        for _ in 0..8 {
            r ^= f(&d);
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / 8.0;
        println!("{name}: {ms:8.2} ms  ({:.1} GB/s) r={r}", n as f64 / ms / 1e6);
    }
}
