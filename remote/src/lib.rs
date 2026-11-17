//! open-cpu-remote: PC・タブレットに接続したスマホ(1台〜複数台)を、ハードウェアアクセラレーターとして使う。
//!
//! 仕組み:
//! - スマホ側で `open-cpu-worker`(aarch64。NEON/dotprod など、その機種で使える命令を open-cpu が自動選択)を動かす。
//! - 手元側(PC・タブレット)の [`Pool`] が、USB の `adb`(`adb forward` 経由)または TCP で各スマホへつなぎ、
//!   行列積などの**計算量が転送量より十分大きい**仕事を、手元のCPUと各スマホへ**実測した速さの比**で振り分ける。
//! - スマホが切断・失敗した場合は、その担当分を手元で計算し直す(結果は常に正しい)。
//!
//! 転送がボトルネックになる仕事(バイト探索のようなメモリ律速のもの)は、手元の AVX-512/AVX2 の方が速いので
//! オフロードしない。使うかどうかは [`Pool::gemm`] が実測値から判断する。

pub mod host;
pub mod kernel;
pub mod protocol;

pub use host::{Node, NodeInfo, Pool};
pub use kernel::{gemm_local, gemm_naive};
