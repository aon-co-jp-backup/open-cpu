//! 手元側: スマホ(worker)への接続、実測による較正、行の振り分け、失敗時の手元での再計算。

use crate::kernel::gemm_local;
use crate::protocol::*;
use std::io;
use std::net::{SocketAddr, TcpStream};
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// 接続先の自己申告(worker の Info 応答)。
#[derive(Clone, Debug, Default)]
pub struct NodeInfo {
    pub arch: String,
    pub threads: usize,
    pub isa: String,
}

/// 較正で得た性能モデル。`1行あたりの時間 = bytes_per_row / bw + flops_per_row / flops`。
#[derive(Clone, Copy, Debug)]
pub struct Perf {
    /// 計算だけの速さ(FLOP/s。worker 自身が測った時間から)。
    pub flops: f64,
    /// 転送の実効速さ(byte/s)。
    pub bw: f64,
    /// 往復の固定遅延(秒)。
    pub latency: f64,
}

pub struct Node {
    pub name: String,
    addr: SocketAddr,
    conn: Mutex<Option<TcpStream>>,
    pub info: NodeInfo,
    pub perf: Perf,
    /// adb 経由なら終了時に `adb forward --remove` する。
    adb_forward: Option<(String, u16)>,
}

fn io_err(m: impl Into<String>) -> io::Error {
    io::Error::other(m.into())
}

impl Node {
    /// TCP で直接つなぐ(`host:port`)。
    pub fn connect_tcp(name: &str, addr: SocketAddr) -> io::Result<Node> {
        let s = TcpStream::connect_timeout(&addr, Duration::from_secs(3))?;
        s.set_nodelay(true).ok();
        s.set_read_timeout(Some(Duration::from_secs(120)))?;
        s.set_write_timeout(Some(Duration::from_secs(120)))?;
        let mut node = Node {
            name: name.to_string(),
            addr,
            conn: Mutex::new(Some(s)),
            info: NodeInfo::default(),
            perf: Perf { flops: 1.0, bw: 1.0, latency: 0.0 },
            adb_forward: None,
        };
        node.info = node.fetch_info()?;
        node.perf = node.calibrate()?;
        Ok(node)
    }

    /// adb の端末(`serial`)の worker(端末側のポート `device_port`)へ `adb forward` でつなぐ。
    pub fn connect_adb(adb: &str, serial: &str, device_port: u16) -> io::Result<Node> {
        let out = Command::new(adb).args(["-s", serial, "forward", "tcp:0", &format!("tcp:{device_port}")]).output()?;
        if !out.status.success() {
            return Err(io_err(format!("adb forward 失敗: {}", String::from_utf8_lossy(&out.stderr).trim())));
        }
        let port: u16 = String::from_utf8_lossy(&out.stdout).trim().parse().map_err(|_| io_err("adb forward のポートを読めない"))?;
        let addr: SocketAddr = ([127, 0, 0, 1], port).into();
        let mut node = Node::connect_tcp(serial, addr).inspect_err(|_| {
            let _ = Command::new(adb).args(["-s", serial, "forward", "--remove", &format!("tcp:{port}")]).output();
        })?;
        node.adb_forward = Some((adb.to_string(), port));
        Ok(node)
    }

    fn call(&self, op: u8, payload: &[u8]) -> io::Result<Vec<u8>> {
        let mut guard = self.conn.lock().map_err(|_| io_err("lock"))?;
        let res = (|| {
            let s = guard.as_mut().ok_or_else(|| io_err("切断済み"))?;
            write_frame(s, op, payload)?;
            let (status, body) = read_frame(s)?;
            if status != 0 {
                return Err(io_err(String::from_utf8_lossy(&body).to_string()));
            }
            Ok(body)
        })();
        if res.is_err() {
            *guard = None; // 壊れた接続は捨てる(以後このノードは使わない)
        }
        res
    }

    pub fn is_alive(&self) -> bool {
        self.conn.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    fn fetch_info(&self) -> io::Result<NodeInfo> {
        let body = self.call(OP_INFO, &[])?;
        let text = String::from_utf8_lossy(&body).to_string();
        let mut info = NodeInfo::default();
        for kv in text.split(';') {
            match kv.split_once('=') {
                Some(("arch", v)) => info.arch = v.to_string(),
                Some(("threads", v)) => info.threads = v.parse().unwrap_or(1),
                Some(("isa", v)) => info.isa = v.to_string(),
                _ => {}
            }
        }
        Ok(info)
    }

    /// 1回の GEMM。戻りは (C, worker内の計算時間[秒])。
    pub fn gemm(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> io::Result<(Vec<f32>, f64)> {
        let body = self.call(OP_GEMM, &encode_gemm(a, b, m, k, n))?;
        if body.len() != 8 + m * n * 4 {
            return Err(io_err("応答の大きさが不正"));
        }
        let ns = u64::from_le_bytes(body[..8].try_into().map_err(|_| io_err("head"))?);
        Ok((bytes_to_f32s(&body[8..]), ns as f64 / 1e9))
    }

    /// 小さい仕事と大きい仕事の2回で、固定遅延・転送速さ・計算の速さを求める。
    fn calibrate(&self) -> io::Result<Perf> {
        let probe = |m: usize, k: usize, n: usize| -> io::Result<(f64, f64, f64)> {
            let a = vec![0.5f32; m * k];
            let b = vec![0.25f32; k * n];
            let t0 = Instant::now();
            let (_, compute) = self.gemm(&a, &b, m, k, n)?;
            let wall = t0.elapsed().as_secs_f64();
            let bytes = ((m * k + k * n + m * n) * 4) as f64;
            Ok((wall, compute, bytes))
        };
        let (w1, _c1, _b1) = probe(8, 8, 8)?; // ほぼ遅延だけ
        let (w2, c2, b2) = probe(256, 256, 256)?;
        let flops = (2.0 * 256.0 * 256.0 * 256.0) / c2.max(1e-6);
        let xfer = (w2 - c2 - w1).max(1e-6);
        Ok(Perf { flops, bw: b2 / xfer, latency: w1 })
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        if let Ok(mut g) = self.conn.lock() {
            if let Some(s) = g.as_mut() {
                let _ = write_frame(s, OP_SHUTDOWN, &[]);
            }
        }
        if let Some((adb, port)) = &self.adb_forward {
            let _ = Command::new(adb).args(["forward", "--remove", &format!("tcp:{port}")]).output();
        }
        let _ = self.addr; // 将来の再接続用に保持
    }
}

/// `adb devices` で接続中の端末の serial 一覧(状態が `device` のもののみ)。
pub fn adb_devices(adb: &str) -> Vec<String> {
    let Ok(out) = Command::new(adb).arg("devices").output() else { return vec![] };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .skip(1)
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            match (it.next(), it.next()) {
                (Some(s), Some("device")) => Some(s.to_string()),
                _ => None,
            }
        })
        .collect()
}

/// 手元のCPU+接続したスマホ群。
pub struct Pool {
    pub nodes: Vec<Node>,
    pub local_threads: usize,
    /// true なら見積もりが得にならなくてもスマホへ振り分ける(通信・正しさの試験用)。
    pub force: bool,
    local_flops: Mutex<Option<f64>>,
}

/// 実行の記録(どこが何行を何秒でやったか)。
#[derive(Debug, Default, Clone)]
pub struct Report {
    pub local_rows: usize,
    pub remote_rows: Vec<(String, usize)>,
    pub fallback_rows: usize,
    pub seconds: f64,
    pub offloaded: bool,
}

impl Pool {
    pub fn new(nodes: Vec<Node>) -> Pool {
        Pool { nodes, local_threads: 0, force: false, local_flops: Mutex::new(None) }
    }

    /// `adb devices` の全端末に、端末側ポート `device_port` の worker があれば接続して加える(つながらない端末は飛ばす)。
    pub fn discover_adb(adb: &str, device_port: u16) -> Pool {
        let nodes = adb_devices(adb).iter().filter_map(|s| Node::connect_adb(adb, s, device_port).ok()).collect();
        Pool::new(nodes)
    }

    fn local_flops(&self) -> f64 {
        let mut g = self.local_flops.lock().unwrap_or_else(|e| e.into_inner());
        *g.get_or_insert_with(|| {
            // 冷えた状態・スレッド起動の影響を避けるため、3回のうち最速を採る。
            let (m, k, n) = (384, 384, 384);
            let (a, b) = (vec![0.5f32; m * k], vec![0.25f32; k * n]);
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                let _ = gemm_local(&a, &b, m, k, n, self.local_threads);
                best = best.min(t.elapsed().as_secs_f64());
            }
            (2.0 * (m * k * n) as f64) / best.max(1e-6)
        })
    }

    /// 1行あたりの見積もり時間(転送+計算)と、固定費(B の転送+遅延)。
    fn node_cost(p: &Perf, k: usize, n: usize) -> (f64, f64) {
        let per_row = ((k + n) * 4) as f64 / p.bw + (2 * k * n) as f64 / p.flops;
        let fixed = p.latency + (k * n * 4) as f64 / p.bw;
        (per_row, fixed)
    }

    /// `C = A×B` を、手元と生きているスマホへ速さの比で振り分けて計算する。
    /// 見積もりでオフロードが得にならない小さい仕事は手元だけで行う。
    pub fn gemm(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, Report) {
        let t0 = Instant::now();
        let mut rep = Report::default();
        let live: Vec<&Node> = self.nodes.iter().filter(|x| x.is_alive()).collect();
        let lf = self.local_flops();
        let l_row = (2 * k * n) as f64 / lf;
        let local_only = l_row * m as f64;

        // 各ノードの 1行時間と固定費。見積もり合計が手元だけより速くなる組み合わせを選ぶ。
        let mut cand: Vec<(&Node, f64, f64)> = live.iter().map(|x| { let (r, f) = Self::node_cost(&x.perf, k, n); (*x, r, f) }).collect();
        cand.sort_by(|x, y| x.1.total_cmp(&y.1));
        let mut chosen: Vec<(&Node, f64, f64)> = vec![];
        let mut best = local_only;
        for i in 1..=cand.len() {
            let set = &cand[..i];
            // 全員が同時に終わる行数配分: rows_j = (T - fixed_j)/row_j、手元は T/l_row。 合計=m を満たす T を解く。
            let (mut s_inv, mut s_fix) = (1.0 / l_row, 0.0);
            for (_, r, f) in set {
                s_inv += 1.0 / r;
                s_fix += f / r;
            }
            let t = (m as f64 + s_fix) / s_inv;
            if (self.force && i == cand.len()) || (t < best * 0.95 && set.iter().all(|(_, _, f)| *f < t)) {
                best = t;
                chosen = set.to_vec();
            }
        }

        if chosen.is_empty() {
            let c = gemm_local(a, b, m, k, n, self.local_threads);
            rep.local_rows = m;
            rep.seconds = t0.elapsed().as_secs_f64();
            return (c, rep);
        }
        rep.offloaded = true;

        // 行の配分。
        let t = best;
        let mut rows: Vec<usize> = if self.force {
            vec![m / (chosen.len() + 1); chosen.len()] // 試験用: 手元とスマホで均等に分ける
        } else {
            chosen.iter().map(|(_, r, f)| (((t - f) / r).floor().max(0.0)) as usize).collect()
        };
        let remote_total: usize = rows.iter().sum();
        let remote_total = remote_total.min(m);
        let local_rows = m - remote_total;
        if remote_total < rows.iter().sum::<usize>() {
            let mut over = rows.iter().sum::<usize>() - remote_total;
            for r in rows.iter_mut().rev() {
                let d = over.min(*r);
                *r -= d;
                over -= d;
            }
        }

        let mut c = vec![0.0f32; m * n];
        let (local_part, rest) = c.split_at_mut(local_rows * n);
        let mut slices: Vec<(&Node, usize, usize, &mut [f32])> = vec![];
        let mut rest = rest;
        let mut start = local_rows;
        for ((node, _, _), &r) in chosen.iter().zip(&rows) {
            let (head, tail) = rest.split_at_mut(r * n);
            slices.push((*node, start, r, head));
            rest = tail;
            start += r;
        }
        let failed = Mutex::new(Vec::<(usize, usize)>::new());
        std::thread::scope(|s| {
            for (node, st, r, out) in slices {
                if r == 0 {
                    continue;
                }
                let failed = &failed;
                s.spawn(move || match node.gemm(&a[st * k..(st + r) * k], b, r, k, n) {
                    Ok((part, _)) => out.copy_from_slice(&part),
                    Err(_) => failed.lock().unwrap_or_else(|e| e.into_inner()).push((st, r)),
                });
            }
            // 手元の担当(最初の local_rows 行)。
            if local_rows > 0 {
                let part = gemm_local(&a[..local_rows * k], b, local_rows, k, n, self.local_threads);
                local_part.copy_from_slice(&part);
            }
        });
        // 失敗した担当分は手元で計算し直す。
        let failed = failed.into_inner().unwrap_or_default();
        for &(st, r) in &failed {
            let part = gemm_local(&a[st * k..(st + r) * k], b, r, k, n, self.local_threads);
            c[st * n..(st + r) * n].copy_from_slice(&part);
            rep.fallback_rows += r;
        }
        rep.local_rows = local_rows;
        for ((node, _, _), &r) in chosen.iter().zip(&rows) {
            rep.remote_rows.push((node.name.clone(), r));
        }
        rep.seconds = t0.elapsed().as_secs_f64();
        (c, rep)
    }
}
