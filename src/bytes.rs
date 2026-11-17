//! バイト列の探索・計数(`memchr` 相当)を、使える最も幅の広い命令で行う。
//!
//! 優先順位: AVX-512BW(64 バイト) > AVX2(32 バイト) > NEON(16 バイト) > スカラー。
//! 使える命令は実行時に [`crate::detect`] で自動判定し(`OPEN_CPU_MAX_ISA` で上限を絞れる)、
//! 呼び出し側は何も指定しなくてよい。結果は常にスカラー版と一致する。

use crate::detect;

/// バイト探索・計数に選ばれる実装。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteImpl {
    Scalar,
    Neon,
    Avx2,
    Avx512,
}

impl std::fmt::Display for ByteImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ByteImpl::Scalar => "scalar",
            ByteImpl::Neon => "neon",
            ByteImpl::Avx2 => "avx2",
            ByteImpl::Avx512 => "avx512bw",
        })
    }
}

/// 現在の CPU でバイト探索・計数に選ばれる実装。
pub fn selected_byte_impl() -> ByteImpl {
    #[cfg(target_arch = "x86_64")]
    {
        let c = detect();
        if c.avx512f && c.avx512bw {
            return ByteImpl::Avx512;
        }
        if c.avx2 {
            return ByteImpl::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return ByteImpl::Neon;
        }
    }
    ByteImpl::Scalar
}

/// `hay` の中で最初に `needle` が現れる位置。
pub fn find_byte(hay: &[u8], needle: u8) -> Option<usize> {
    match selected_byte_impl() {
        #[cfg(target_arch = "x86_64")]
        ByteImpl::Avx512 => unsafe { find_avx512(hay, needle) },
        #[cfg(target_arch = "x86_64")]
        ByteImpl::Avx2 => unsafe { find_avx2(hay, needle) },
        #[cfg(target_arch = "aarch64")]
        ByteImpl::Neon => unsafe { find_neon(hay, needle) },
        _ => find_byte_scalar(hay, needle),
    }
}

/// `hay` の中の `needle` の個数。
pub fn count_byte(hay: &[u8], needle: u8) -> usize {
    match selected_byte_impl() {
        #[cfg(target_arch = "x86_64")]
        ByteImpl::Avx512 => unsafe { count_avx512(hay, needle) },
        #[cfg(target_arch = "x86_64")]
        ByteImpl::Avx2 => unsafe { count_avx2(hay, needle) },
        #[cfg(target_arch = "aarch64")]
        ByteImpl::Neon => unsafe { count_neon(hay, needle) },
        _ => count_byte_scalar(hay, needle),
    }
}

/// [`find_byte`] のスカラー参照実装。
pub fn find_byte_scalar(hay: &[u8], needle: u8) -> Option<usize> {
    hay.iter().position(|&b| b == needle)
}

/// [`count_byte`] のスカラー参照実装。
pub fn count_byte_scalar(hay: &[u8], needle: u8) -> usize {
    hay.iter().filter(|&&b| b == needle).count()
}

// ---------------------------------------------------------------------------
// x86_64
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn find_avx512(hay: &[u8], needle: u8) -> Option<usize> {
    use std::arch::x86_64::*;
    unsafe {
        let n = hay.len();
        let v = _mm512_set1_epi8(needle as i8);
        let mut i = 0;
        while i + 64 <= n {
            let x = _mm512_loadu_si512(hay.as_ptr().add(i) as *const _);
            let m = _mm512_cmpeq_epi8_mask(x, v);
            if m != 0 {
                return Some(i + m.trailing_zeros() as usize);
            }
            i += 64;
        }
        find_byte_scalar(&hay[i..], needle).map(|p| i + p)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn count_avx512(hay: &[u8], needle: u8) -> usize {
    use std::arch::x86_64::*;
    unsafe {
        let n = hay.len();
        let v = _mm512_set1_epi8(needle as i8);
        let mut total = 0usize;
        let mut i = 0;
        while i + 64 <= n {
            let x = _mm512_loadu_si512(hay.as_ptr().add(i) as *const _);
            total += _mm512_cmpeq_epi8_mask(x, v).count_ones() as usize;
            i += 64;
        }
        total + count_byte_scalar(&hay[i..], needle)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_avx2(hay: &[u8], needle: u8) -> Option<usize> {
    use std::arch::x86_64::*;
    unsafe {
        let n = hay.len();
        let v = _mm256_set1_epi8(needle as i8);
        let mut i = 0;
        while i + 32 <= n {
            let x = _mm256_loadu_si256(hay.as_ptr().add(i) as *const __m256i);
            let m = _mm256_movemask_epi8(_mm256_cmpeq_epi8(x, v)) as u32;
            if m != 0 {
                return Some(i + m.trailing_zeros() as usize);
            }
            i += 32;
        }
        find_byte_scalar(&hay[i..], needle).map(|p| i + p)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn count_avx2(hay: &[u8], needle: u8) -> usize {
    use std::arch::x86_64::*;
    unsafe {
        let n = hay.len();
        let v = _mm256_set1_epi8(needle as i8);
        let mut total = 0usize;
        let mut i = 0;
        while i + 32 <= n {
            let x = _mm256_loadu_si256(hay.as_ptr().add(i) as *const __m256i);
            let m = _mm256_movemask_epi8(_mm256_cmpeq_epi8(x, v)) as u32;
            total += m.count_ones() as usize;
            i += 32;
        }
        total + count_byte_scalar(&hay[i..], needle)
    }
}

// ---------------------------------------------------------------------------
// aarch64
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn find_neon(hay: &[u8], needle: u8) -> Option<usize> {
    use std::arch::aarch64::*;
    unsafe {
        let n = hay.len();
        let v = vdupq_n_u8(needle);
        let mut i = 0;
        while i + 16 <= n {
            let x = vld1q_u8(hay.as_ptr().add(i));
            let eq = vceqq_u8(x, v);
            if vmaxvq_u8(eq) != 0 {
                // 16 バイトのどこか: スカラーで位置を決める(ブロック内のみ)
                return find_byte_scalar(&hay[i..i + 16], needle).map(|p| i + p);
            }
            i += 16;
        }
        find_byte_scalar(&hay[i..], needle).map(|p| i + p)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn count_neon(hay: &[u8], needle: u8) -> usize {
    use std::arch::aarch64::*;
    unsafe {
        let n = hay.len();
        let v = vdupq_n_u8(needle);
        let mut total = 0usize;
        let mut i = 0;
        while i + 16 <= n {
            let x = vld1q_u8(hay.as_ptr().add(i));
            // 一致したレーンは 0xFF → 右に 7 ビットずらして 1 にし、水平加算
            total += vaddvq_u8(vshrq_n_u8::<7>(vceqq_u8(x, v))) as usize;
            i += 16;
        }
        total + count_byte_scalar(&hay[i..], needle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(len: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 7) as u8 // 小さな値域 → 一致が頻繁に起きる
            })
            .collect()
    }

    #[test]
    fn matches_scalar_for_all_lengths_and_needles() {
        for len in 0..300 {
            let d = data(len, len as u64 + 11);
            for needle in 0..9u8 {
                assert_eq!(find_byte(&d, needle), find_byte_scalar(&d, needle), "find len={len} needle={needle}");
                assert_eq!(count_byte(&d, needle), count_byte_scalar(&d, needle), "count len={len} needle={needle}");
            }
        }
    }

    #[test]
    fn finds_at_every_position_and_edges() {
        for len in [1usize, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000] {
            for pos in 0..len {
                let mut d = vec![0u8; len];
                d[pos] = 9;
                assert_eq!(find_byte(&d, 9), Some(pos), "len={len} pos={pos}");
                assert_eq!(count_byte(&d, 9), 1);
            }
            assert_eq!(find_byte(&vec![0u8; len], 9), None);
        }
    }

    #[test]
    fn high_bytes_and_empty() {
        assert_eq!(find_byte(&[], 1), None);
        assert_eq!(count_byte(&[], 1), 0);
        let d = vec![0xFFu8; 200];
        assert_eq!(count_byte(&d, 0xFF), 200);
        assert_eq!(find_byte(&d, 0x80), None);
    }

    #[test]
    fn impl_is_reported() {
        let s = selected_byte_impl().to_string();
        assert!(["scalar", "neon", "avx2", "avx512bw"].contains(&s.as_str()));
    }
}
