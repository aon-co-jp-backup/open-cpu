//! `cargo run --release --example remote_bench -- [m k n] [--tcp host:port]`
//! adb でつないだ端末の worker(端末側ポート 7878)を自動で探して使う。

use open_cpu_remote::{gemm_local, host::Node, Pool};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let nums: Vec<usize> = args.iter().filter_map(|a| a.parse().ok()).collect();
    let (m, k, n) = (nums.first().copied().unwrap_or(1024), nums.get(1).copied().unwrap_or(512), nums.get(2).copied().unwrap_or(512));
    let adb = std::env::var("ADB").unwrap_or_else(|_| "adb".into());
    let pool = if let Some(pos) = args.iter().position(|a| a == "--tcp") {
        let addr = args[pos + 1].parse().expect("host:port");
        Pool::new(vec![Node::connect_tcp("tcp", addr).expect("connect")])
    } else {
        Pool::discover_adb(&adb, 7878)
    };
    let mut pool = pool;
    pool.force = args.iter().any(|a| a == "--force");
    for nd in &pool.nodes {
        println!("node {}: {} threads={} isa={} compute={:.1} GFLOP/s bw={:.0} MB/s latency={:.1} ms", nd.name, nd.info.arch, nd.info.threads, nd.info.isa, nd.perf.flops / 1e9, nd.perf.bw / 1e6, nd.perf.latency * 1e3);
    }
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) / 7.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) / 9.0).collect();
    let t = std::time::Instant::now();
    let want = gemm_local(&a, &b, m, k, n, 0);
    let local = t.elapsed().as_secs_f64();
    let (got, rep) = pool.gemm(&a, &b, m, k, n);
    let maxerr = got.iter().zip(&want).map(|(x, y)| (x - y).abs() / (1.0 + y.abs())).fold(0.0f32, f32::max);
    let gf = |s: f64| 2.0 * (m * k * n) as f64 / s / 1e9;
    println!("{m}x{k}x{n}: 手元のみ {:.3}s ({:.1} GFLOP/s) / プール {:.3}s ({:.1} GFLOP/s) offload={} local_rows={} remote={:?} fallback={} 最大相対誤差={:.2e}", local, gf(local), rep.seconds, gf(rep.seconds), rep.offloaded, rep.local_rows, rep.remote_rows, rep.fallback_rows, maxerr);
}
