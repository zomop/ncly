//! NCLY micro-benchmark runner.
//!
//! Why not `criterion`? A benchmark harness that cannot be rebuilt on the target
//! machine is not reproducible, and NCLY's CI does not vendor dependencies for
//! the benchmark profile. This runner is deliberately small, has no dependencies
//! beyond the workspace, and reports the same statistics that matter:
//!
//! * median wall time per iteration (robust to scheduler noise),
//! * best-of-N time (the "uncontended" latency),
//! * effective throughput in GFLOP/s where the operation count is known.
//!
//! # Method
//!
//! Each case runs a warm-up pass, then `iterations` timed repetitions. Memory is
//! allocated outside the timed region. Results are printed as a Markdown table so
//! they can be pasted into `BENCHMARKS.md` verbatim — NCLY never publishes a
//! number that was not produced by this program on the machine that reports it.
//!
//! ```text
//! cargo run --release -p ncly-benchmarks -- [filter]
//! ```

use ncly_linalg::lu::Lu;
use ncly_sparse::CsrMatrix;
use ncly_tensor::Tensor;
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Collected measurement for one case.
struct Measurement {
    name: &'static str,
    size: String,
    median: Duration,
    best: Duration,
    flops: Option<f64>,
    extra: String,
}

impl Measurement {
    fn gflops(&self) -> Option<f64> {
        let flops = self.flops?;
        Some(flops / self.median.as_secs_f64() / 1e9)
    }

    fn gib_per_second(&self) -> Option<f64> {
        // Bytes moved are stored in `extra` for memory-bound cases by appending
        // "; bytes=<n>".
        let marker = "bytes=";
        let start = self.extra.find(marker)? + marker.len();
        let rest = &self.extra[start..];
        let end = rest.find(';').unwrap_or(rest.len());
        let bytes: f64 = rest[..end].parse().ok()?;
        Some(bytes / self.median.as_secs_f64() / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Run `f` repeatedly and record the median and best durations.
fn measure<F: FnMut() -> R>(
    name: &'static str,
    size: String,
    iterations: usize,
    mut f: F,
) -> Measurement
where
    R: Sized,
{
    // Warm-up: page in memory and let the branch predictors settle.
    for _ in 0..(iterations / 10).max(1) {
        black_box(f());
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        black_box(result);
        samples.push(elapsed);
    }
    samples.sort_unstable();
    Measurement {
        name,
        size,
        median: samples[samples.len() / 2],
        best: samples[0],
        flops: None,
        extra: String::new(),
    }
}

fn with_flops(mut measurement: Measurement, flops: f64) -> Measurement {
    measurement.flops = Some(flops);
    measurement
}

fn with_bytes(mut measurement: Measurement, bytes: usize) -> Measurement {
    measurement.extra = format!("bytes={bytes}");
    measurement
}

fn main() {
    let filter: Option<String> = std::env::args().nth(1);
    let mut results: Vec<Measurement> = Vec::new();
    let want = |name: &str| match &filter {
        Some(pattern) => name.contains(pattern.as_str()),
        None => true,
    };

    println!("# NCLY benchmark results");
    println!();
    println!("kernel: `{}`", ncly_simd::active_kernel().name());
    println!("rustc: {}", rustc_version());
    println!();

    if want("axpy") {
        results.push(benchmark_axpy());
    }
    if want("add") {
        results.push(benchmark_elementwise_add());
    }
    if want("exp") {
        results.push(benchmark_exp());
    }
    if want("sum") {
        results.push(benchmark_sum());
    }
    if want("matmul") {
        results.extend(benchmark_matmul());
    }
    if want("lu") {
        results.push(benchmark_lu());
    }
    if want("spmv") {
        results.push(benchmark_spmv());
    }
    if want("grad") {
        results.push(benchmark_autodiff());
    }
    if want("transpose") {
        results.push(benchmark_transpose_copy());
    }

    print_table(&results);
}

fn rustc_version() -> String {
    option_env!("RUSTC_VERSION")
        .map(str::to_string)
        .unwrap_or_else(|| format!("(build of ncly {})", env!("CARGO_PKG_VERSION")))
}

fn benchmark_axpy() -> Measurement {
    let n = 1 << 20; // 8 MiB per array, larger than L2
    let a = vec![1.5f64; n];
    let b = vec![0.5f64; n];
    let mut out = vec![0.0f64; n];
    measure("axpy (y = 2a + b)", format!("n=2^20"), 50, || {
        ncly_simd::axpy(&mut out, 2.0, &a, 1, &b, 1);
        out[0]
    })
    .pipe(|m| with_bytes(with_flops(m, 3.0 * n as f64), 3 * n * 8))
}

fn benchmark_elementwise_add() -> Measurement {
    let n = 1 << 20;
    let a = Tensor::uniform([n], 1);
    let b = Tensor::uniform([n], 2);
    measure("tensor add (1-d)", format!("n=2^20"), 50, || {
        a.add(&b).map(|t| t.sum())
    })
    .pipe(|m| with_bytes(with_flops(m, n as f64), 3 * n * 8))
}

fn benchmark_exp() -> Measurement {
    let n = 1 << 20;
    let a = Tensor::uniform([n], 3);
    measure("exp (1-d)", format!("n=2^20"), 30, || a.exp().sum())
        .pipe(|m| with_bytes(with_flops(m, 8.0 * n as f64), 2 * n * 8))
}

fn benchmark_sum() -> Measurement {
    let n = 1 << 20;
    let a = Tensor::uniform([n], 4);
    measure("sum (1-d)", format!("n=2^20"), 50, || a.sum())
        .pipe(|m| with_bytes(with_flops(m, n as f64), n * 8))
}

fn benchmark_matmul() -> Vec<Measurement> {
    let mut out = Vec::new();
    for n in [64usize, 128, 256, 512] {
        let a = Tensor::uniform([n, n], n as u64);
        let b = Tensor::uniform([n, n], n as u64 + 1);
        let iterations = match n {
            64 => 200,
            128 => 80,
            256 => 20,
            _ => 5,
        };
        // 2n³ flops per product (n³ multiplies, n³ additions).
        let flops = 2.0 * (n as f64).powi(3);
        out.push(
            measure(
                "matmul (f64, square)",
                format!("{n}x{n}"),
                iterations,
                || a.matmul(&b).map(|t| t.sum()),
            )
            .pipe(|m| with_flops(m, flops)),
        );
    }
    out
}

fn benchmark_lu() -> Measurement {
    let n = 256;
    let data: Vec<f64> = (0..n * n)
        .map(|i| {
            let (row, col) = (i / n, i % n);
            if row == col {
                n as f64 * 2.0
            } else {
                ((i * 7919) % 1000) as f64 / 1000.0 - 0.5
            }
        })
        .collect();
    // 2n³/3 flops for the elimination.
    let flops = 2.0 / 3.0 * (n as f64).powi(3);
    measure("LU factorisation", format!("{n}x{n}"), 20, || {
        Lu::factor(&data, n).map(|lu| lu.determinant())
    })
    .pipe(|m| with_flops(m, flops))
}

fn benchmark_spmv() -> Measurement {
    let n = 200_000;
    let mut row = Vec::new();
    let mut col = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        row.push(i);
        col.push(i);
        values.push(2.0);
        if i + 1 < n {
            row.push(i);
            col.push(i + 1);
            values.push(-1.0);
        }
    }
    let matrix = CsrMatrix::from_triplets(n, n, &row, &col, &values).expect("valid triplets");
    let x = vec![1.0f64; n];
    let nnz = matrix.nnz();
    // 2 flops per stored entry.
    let flops = 2.0 * nnz as f64;
    let bytes = nnz * (8 + 8) + n * 8 * 2;
    measure(
        "spmv (1-d Laplacian)",
        format!("n={n}, nnz={nnz}"),
        50,
        || matrix.spmv(&x).iter().sum::<f64>(),
    )
    .pipe(|m| with_bytes(with_flops(m, flops), bytes))
}

fn benchmark_autodiff() -> Measurement {
    // Forward and backward through a small chain over a 4096-element tensor.
    let n = 4096;
    let tape = ncly_autodiff::Tape::new();
    let _guard = ncly_autodiff::Variable::with_tape(&tape);
    let x = ncly_autodiff::Variable::auto_tensor(Tensor::uniform([n], 9), "x");
    measure("autodiff fwd+bwd (exp·sin)", format!("n={n}"), 30, || {
        x.zero_grad();
        let y = x.exp().mul(&x.sin()).mean();
        y.backward();
        y.scalar().unwrap_or(0.0)
    })
    .pipe(|m| with_flops(m, 8.0 * n as f64))
}

fn benchmark_transpose_copy() -> Measurement {
    let n = 1024;
    let a = Tensor::uniform([n, n], 12);
    let bytes = n * n * 8 * 2;
    measure("transpose + materialise", format!("{n}x{n}"), 10, || {
        a.transpose().to_contiguous().sum()
    })
    .pipe(|m| with_bytes(m, bytes))
}

fn print_table(results: &[Measurement]) {
    println!("| kernel | size | median | best | GFLOP/s | GiB/s |");
    println!("| --- | --- | ---: | ---: | ---: | ---: |");
    for measurement in results {
        let gflops = measurement
            .gflops()
            .map(|g| format!("{g:.2}"))
            .unwrap_or_else(|| "—".to_string());
        let bandwidth = measurement
            .gib_per_second()
            .map(|g| format!("{g:.2}"))
            .unwrap_or_else(|| "—".to_string());
        println!(
            "| {} | {} | {:.3} ms | {:.3} ms | {} | {} |",
            measurement.name,
            measurement.size,
            measurement.median.as_secs_f64() * 1e3,
            measurement.best.as_secs_f64() * 1e3,
            gflops,
            bandwidth
        );
    }
    println!();
    println!("Median of N repetitions after a 10% warm-up. Single-threaded.");
    println!("Numbers are only meaningful relative to the machine that produced them.");
}

/// Small helper so benchmarks can be written as a pipeline.
trait Pipe: Sized {
    fn pipe<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }
}

impl<T> Pipe for T {}
