//! 同一プロセス内の模擬 worker で、プロトコル・振り分け・失敗時の手元再計算を確かめる。

use open_cpu_remote::host::Node;
use open_cpu_remote::kernel::{gemm_local, gemm_naive};
use open_cpu_remote::protocol::*;
use open_cpu_remote::Pool;
use std::net::{SocketAddr, TcpListener};

/// `fail_after` 回目の GEMM 要求で接続を切る worker(0 なら切らない)。
fn spawn_worker(fail_after: usize) -> SocketAddr {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    std::thread::spawn(move || {
        for conn in l.incoming() {
            let Ok(mut s) = conn else { continue };
            let mut gemms = 0;
            loop {
                let Ok((op, p)) = read_frame(&mut s) else { break };
                match op {
                    OP_INFO => write_frame(&mut s, 0, b"arch=test;threads=2;isa=scalar").unwrap(),
                    OP_GEMM => {
                        gemms += 1;
                        if fail_after > 0 && gemms >= fail_after {
                            break; // 切断
                        }
                        let (a, b, m, k, n) = decode_gemm(&p).unwrap();
                        let c = gemm_local(&a, &b, m, k, n, 0);
                        let mut out = 1_000u64.to_le_bytes().to_vec();
                        f32s_to_bytes(&c, &mut out);
                        write_frame(&mut s, 0, &out).unwrap();
                    }
                    _ => break,
                }
            }
        }
    });
    addr
}

fn data(m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    ((0..m * k).map(|i| ((i % 13) as f32 - 6.0) / 7.0).collect(), (0..k * n).map(|i| ((i % 11) as f32 - 5.0) / 9.0).collect())
}

fn close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (x, y) in got.iter().zip(want) {
        assert!((x - y).abs() <= 1e-2 * (1.0 + y.abs()), "{x} vs {y}");
    }
}

#[test]
fn node_gemm_matches_reference() {
    let node = Node::connect_tcp("t", spawn_worker(0)).unwrap();
    assert_eq!(node.info.threads, 2);
    let (a, b) = data(37, 29, 41);
    let (c, _) = node.gemm(&a, &b, 37, 29, 41).unwrap();
    close(&c, &gemm_naive(&a, &b, 37, 29, 41));
}

#[test]
fn pool_result_is_correct_with_two_workers() {
    let pool = Pool::new(vec![Node::connect_tcp("a", spawn_worker(0)).unwrap(), Node::connect_tcp("b", spawn_worker(0)).unwrap()]);
    for &(m, k, n) in &[(5, 7, 3), (300, 300, 300), (513, 128, 257)] {
        let (a, b) = data(m, k, n);
        let (c, rep) = pool.gemm(&a, &b, m, k, n);
        close(&c, &gemm_naive(&a, &b, m, k, n));
        assert_eq!(rep.local_rows + rep.remote_rows.iter().map(|x| x.1).sum::<usize>(), if rep.offloaded { m } else { m });
    }
}

#[test]
fn worker_failure_falls_back_to_local_and_stays_correct() {
    // 較正の2回(小・大)は通し、その後の要求で切断する worker。
    let pool = Pool::new(vec![Node::connect_tcp("flaky", spawn_worker(3)).unwrap()]);
    let (m, k, n) = (600, 400, 400);
    let (a, b) = data(m, k, n);
    let (c, rep) = pool.gemm(&a, &b, m, k, n);
    close(&c, &gemm_naive(&a, &b, m, k, n));
    if rep.offloaded {
        assert!(rep.fallback_rows > 0, "切断されたのに手元で再計算されていない: {rep:?}");
        let (c2, rep2) = pool.gemm(&a, &b, m, k, n);
        close(&c2, &gemm_naive(&a, &b, m, k, n));
        assert!(!rep2.offloaded, "切断済みノードを再び使った");
    }
}

#[test]
fn empty_pool_works_locally() {
    let pool = Pool::new(vec![]);
    let (a, b) = data(10, 10, 10);
    let (c, rep) = pool.gemm(&a, &b, 10, 10, 10);
    close(&c, &gemm_naive(&a, &b, 10, 10, 10));
    assert!(!rep.offloaded);
}
