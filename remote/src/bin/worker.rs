//! スマホ(や別のPC)で動かす計算ワーカー。`open-cpu-worker [ポート(既定 7878)] [--bind アドレス]`
//!
//! 既定は 127.0.0.1 のみで待ち受ける(adb forward 経由で使う)。LAN に開くときは `--bind 0.0.0.0` を明示する。
//! 認証は無いので、信頼できるネットワークでのみ使うこと。

use open_cpu_remote::kernel::gemm_local;
use open_cpu_remote::protocol::*;
use std::io::{self, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Instant;

fn info() -> String {
    format!(
        "arch={};threads={};isa={}",
        std::env::consts::ARCH,
        std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1),
        open_cpu::selected_float_impl()
    )
}

fn serve(mut s: TcpStream) -> io::Result<bool> {
    s.set_nodelay(true).ok();
    loop {
        let (op, payload) = match read_frame(&mut s) {
            Ok(x) => x,
            Err(_) => return Ok(false), // 切断
        };
        match op {
            OP_INFO => write_frame(&mut s, 0, info().as_bytes())?,
            OP_GEMM => match decode_gemm(&payload) {
                Ok((a, b, m, k, n)) => {
                    let t = Instant::now();
                    let c = gemm_local(&a, &b, m, k, n, 0);
                    let ns = t.elapsed().as_nanos() as u64;
                    let mut out = Vec::with_capacity(8 + c.len() * 4);
                    out.extend_from_slice(&ns.to_le_bytes());
                    f32s_to_bytes(&c, &mut out);
                    write_frame(&mut s, 0, &out)?;
                }
                Err(e) => write_frame(&mut s, 1, e.as_bytes())?,
            },
            OP_SHUTDOWN => return Ok(false),
            _ => write_frame(&mut s, 1, b"unknown op")?,
        }
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut port: u16 = 7878;
    let mut bind = "127.0.0.1".to_string();
    let mut once = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--bind" => {
                i += 1;
                bind = args.get(i).cloned().unwrap_or(bind);
            }
            "--once" => once = true,
            p => port = p.parse().unwrap_or(port),
        }
        i += 1;
    }
    let l = TcpListener::bind((bind.as_str(), port))?;
    println!("open-cpu-worker listening on {}:{} ({})", bind, port, info());
    io::stdout().flush().ok();
    for conn in l.incoming() {
        let Ok(s) = conn else { continue };
        // 1接続ずつ順に処理(計算が全コアを使うため並列にしない)。
        let _ = serve(s);
        if once {
            break;
        }
    }
    Ok(())
}
