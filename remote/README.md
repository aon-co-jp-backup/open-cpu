# open-cpu-remote — スマホをハードウェアアクセラレーターにする(open-cpu の拡張)

PC・タブレットに**スマホを1台〜複数台**つなぎ(USB の `adb`、または TCP)、行列積(f32 GEMM)を手元のCPUと各スマホへ**実測した速さの比**で振り分けます。スマホが切れたら担当分を手元で計算し直すので、結果は常に正しいです。

```bash
# スマホ側(aarch64-linux-android 向けにビルドして adb push、127.0.0.1:7878 で待ち受け)
cargo build --release --target aarch64-linux-android --bin open-cpu-worker
adb push target/aarch64-linux-android/release/open-cpu-worker /data/local/tmp/ && adb shell "nohup /data/local/tmp/open-cpu-worker 7878 &"
# 手元側: 接続中の全端末を自動で探して使う
cargo run --release --example remote_bench -- 2048 2048 2048     # --force で見積もりに関係なく分担(試験用)
```

```rust
let pool = open_cpu_remote::Pool::discover_adb("adb", 7878);
let (c, report) = pool.gemm(&a, &b, m, k, n);   // report に各担当の行数・フォールバック行数
```

## 実測(Snapdragon 端末 6スレッド NEON ↔ 開発PC AVX2)と正直な結論
- 較正値: スマホの計算 約 2〜4 GFLOP/s、`adb forward` の実効転送 約 10〜20 MB/s、往復遅延 約 7〜9 ms。手元PCは約 200〜270 GFLOP/s。
- 2台同時(moto g53y SM4350 8スレッド: 計算 約9 GFLOP/s、Snapdragon M06 SM7435 6スレッド: 約4 GFLOP/s)でも、強制分担(手元342行+各341行)の結果は手元のみと一致(最大相対誤差0)。ただし転送が約16〜20 MB/s のため遅くなる(0.008 s → 0.39 s)。
- そのため**この組み合わせでは自動判断が「手元のみ」を選ぶ**(強制すると 手元のみ 0.003 s → 0.15 s と遅くなる)。結果の一致(最大相対誤差 0)と切断時のフォールバックは確認済み。
- 効くのは「手元のCPUが非力(古いタブレット・低電力機)」「計算量が転送量より桁違いに大きい」「スマホが高性能で多台数」のとき。転送がボトルネックなので、バイト探索のようなメモリ律速の処理はオフロードしない。
- 認証は無い。worker は既定で 127.0.0.1 のみ。`--bind 0.0.0.0` は信頼できるネットワークでのみ。

## 試験
`cargo test --release`(同一プロセス内の模擬 worker で、結果の一致・2台振り分け・切断時の手元再計算)。実機は上記 `remote_bench`(Snapdragon 端末で確認、2026-11-17)。
