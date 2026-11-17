//! 手元とスマホ(worker)の間の通信。長さ付きのバイナリ。
//!
//! 要求: `[op: u8][len: u64 LE][payload]`、応答: `[status: u8 (0=成功, 1=失敗)][len: u64 LE][payload]`。
//! f32 は小文字エンディアン(x86_64・aarch64 とも)。1回の最大は [`MAX_FRAME`] バイト。

use std::io::{self, Read, Write};

pub const OP_INFO: u8 = 1;
pub const OP_GEMM: u8 = 2;
pub const OP_SHUTDOWN: u8 = 3;

/// 1フレームの最大(攻撃的な長さの指定でメモリを使い切らないための上限): 1 GiB。
pub const MAX_FRAME: u64 = 1 << 30;

pub fn write_frame(w: &mut impl Write, tag: u8, payload: &[u8]) -> io::Result<()> {
    w.write_all(&[tag])?;
    w.write_all(&(payload.len() as u64).to_le_bytes())?;
    w.write_all(payload)?;
    w.flush()
}

pub fn read_frame(r: &mut impl Read) -> io::Result<(u8, Vec<u8>)> {
    let mut head = [0u8; 9];
    r.read_exact(&mut head)?;
    let len = u64::from_le_bytes(head[1..9].try_into().unwrap_or([0; 8]));
    if len > MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "frame too large"));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    Ok((head[0], buf))
}

pub fn f32s_to_bytes(v: &[f32], out: &mut Vec<u8>) {
    out.reserve(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

pub fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// GEMM 要求の本文: `m,k,n (u32 LE ×3) + A(m*k) + B(k*n)`。
pub fn encode_gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<u8> {
    let mut p = Vec::with_capacity(12 + (a.len() + b.len()) * 4);
    for v in [m, k, n] {
        p.extend_from_slice(&(v as u32).to_le_bytes());
    }
    f32s_to_bytes(a, &mut p);
    f32s_to_bytes(b, &mut p);
    p
}

pub fn decode_gemm(p: &[u8]) -> Result<(Vec<f32>, Vec<f32>, usize, usize, usize), String> {
    if p.len() < 12 {
        return Err("short gemm payload".into());
    }
    let rd = |o: usize| u32::from_le_bytes([p[o], p[o + 1], p[o + 2], p[o + 3]]) as usize;
    let (m, k, n) = (rd(0), rd(4), rd(8));
    let (na, nb) = (m.checked_mul(k).ok_or("overflow")?, k.checked_mul(n).ok_or("overflow")?);
    let need = 12usize.checked_add(na.checked_add(nb).ok_or("overflow")?.checked_mul(4).ok_or("overflow")?).ok_or("overflow")?;
    if p.len() != need {
        return Err(format!("gemm payload size {} != {}", p.len(), need));
    }
    let a = bytes_to_f32s(&p[12..12 + na * 4]);
    let b = bytes_to_f32s(&p[12 + na * 4..]);
    Ok((a, b, m, k, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_round_trip() {
        let mut buf = Vec::new();
        write_frame(&mut buf, 7, b"hello").unwrap();
        let (t, p) = read_frame(&mut &buf[..]).unwrap();
        assert_eq!((t, p.as_slice()), (7, &b"hello"[..]));
    }

    #[test]
    fn rejects_huge_length_without_allocating() {
        let mut head = vec![1u8];
        head.extend_from_slice(&(MAX_FRAME + 1).to_le_bytes());
        assert!(read_frame(&mut &head[..]).is_err());
    }

    #[test]
    fn gemm_payload_round_trip_and_bad_sizes() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0; 6];
        let p = encode_gemm(&a, &b, 2, 3, 2);
        let (a2, b2, m, k, n) = decode_gemm(&p).unwrap();
        assert_eq!((a2, b2, m, k, n), (a, b, 2, 3, 2));
        assert!(decode_gemm(&p[..p.len() - 1]).is_err());
        assert!(decode_gemm(&[0u8; 5]).is_err());
        let mut huge = vec![0u8; 12];
        huge[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        huge[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_gemm(&huge).is_err());
    }
}
