//! Tensor engine validation against independently computed references.

use ncly_integration_tests::{assert_close, max_abs_diff, Rng};
use ncly_tensor::Tensor;

/// Elementwise reference computed with plain loops.
fn reference_broadcast(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
    out_shape: &[usize],
    op: impl Fn(f64, f64) -> f64,
) -> Vec<f64> {
    let size: usize = out_shape.iter().product();
    let offset = |index: &[usize], shape: &[usize]| -> usize {
        let pad = out_shape.len() - shape.len();
        let mut flat = 0usize;
        let mut stride = 1usize;
        for axis in (0..shape.len()).rev() {
            let out_axis = axis + pad;
            let i = if shape[axis] == 1 { 0 } else { index[out_axis] };
            flat += i * stride;
            stride *= shape[axis];
        }
        flat
    };
    let mut out = Vec::with_capacity(size);
    let mut index = vec![0usize; out_shape.len()];
    for _ in 0..size {
        let ai = offset(&index, a_shape);
        let bi = offset(&index, b_shape);
        out.push(op(a[ai], b[bi]));
        for axis in (0..out_shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < out_shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    out
}

#[test]
fn broadcasting_matches_independent_reference() {
    let cases: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![3], vec![3]),
        (vec![2, 3], vec![3]),
        (vec![2, 1], vec![2, 3]),
        (vec![1, 3, 1], vec![2, 1, 4]),
        (vec![2, 3, 4], vec![4]),
        (vec![], vec![2, 2]),
        (vec![2, 1, 3], vec![1, 4, 1]),
    ];
    let mut rng = Rng::new(11);
    for (a_shape, b_shape) in cases {
        let a_size: usize = a_shape.iter().product();
        let b_size: usize = b_shape.iter().product();
        let a_values: Vec<f64> = (0..a_size).map(|_| rng.range(-2.0, 2.0)).collect();
        let b_values: Vec<f64> = (0..b_size).map(|_| rng.range(-2.0, 2.0)).collect();
        let a = Tensor::from_vec(a_values.clone(), a_shape.clone()).unwrap();
        let b = Tensor::from_vec(b_values.clone(), b_shape.clone()).unwrap();
        let out_shape = a.shape().broadcast_with(b.shape()).unwrap();
        for (name, got, op) in [
            (
                "add",
                a.add(&b).unwrap(),
                (|x: f64, y: f64| x + y) as fn(f64, f64) -> f64,
            ),
            ("sub", a.sub(&b).unwrap(), |x, y| x - y),
            ("mul", a.mul(&b).unwrap(), |x, y| x * y),
        ] {
            let expected = reference_broadcast(
                &a_values,
                &a_shape,
                &b_values,
                &b_shape,
                out_shape.dims(),
                op,
            );
            assert_close(
                &got.to_vec(),
                &expected,
                1e-12,
                &format!("{name} {a_shape:?} x {b_shape:?}"),
            );
        }
    }
}

#[test]
fn transposed_and_broadcast_operands_agree_with_materialised_versions() {
    let mut rng = Rng::new(23);
    let values: Vec<f64> = (0..12).map(|_| rng.range(-1.0, 1.0)).collect();
    let base = Tensor::from_vec(values, [3, 4]).unwrap();
    let view = base.transpose(); // (4, 3)
    let row = Tensor::from_vec(vec![0.5, -0.5, 1.0], [3]).unwrap();

    let from_view = view.mul(&row).unwrap();
    let from_dense = view.to_contiguous().mul(&row).unwrap();
    assert_eq!(from_view.to_vec(), from_dense.to_vec());

    // Reduction over a transposed view equals reduction over its materialisation.
    assert!((view.sum() - base.sum()).abs() < 1e-12);
    let axis0_view = view.sum_axis(0).unwrap().to_vec();
    let axis0_dense = view.to_contiguous().sum_axis(0).unwrap().to_vec();
    assert_close(&axis0_view, &axis0_dense, 1e-12, "axis-0 reduction of a view");
}

#[test]
fn reductions_match_dense_reference() {
    let mut rng = Rng::new(31);
    let shape = [4usize, 5, 3];
    let size: usize = shape.iter().product();
    let values: Vec<f64> = (0..size).map(|_| rng.range(-3.0, 3.0)).collect();
    let tensor = Tensor::from_vec(values.clone(), shape).unwrap();

    // Total sum with Neumaier compensation as the reference.
    let mut total = 0.0;
    let mut compensation = 0.0;
    for &value in &values {
        let t = total + value;
        if total.abs() >= value.abs() {
            compensation += (total - t) + value;
        } else {
            compensation += (value - t) + total;
        }
        total = t;
    }
    assert!((tensor.sum() - (total + compensation)).abs() < 1e-10);

    // Axis 1 sums.
    let reduced = tensor.sum_axis(1).unwrap();
    assert_eq!(reduced.shape().dims(), &[4, 3]);
    for i in 0..4 {
        for k in 0..3 {
            let mut expected = 0.0;
            for j in 0..5 {
                expected += tensor.get(&[i, j, k]).unwrap();
            }
            let got = reduced.get(&[i, k]).unwrap();
            assert!((got - expected).abs() < 1e-12, "sum axis 1 at ({i},{k})");
        }
    }
}

#[test]
fn matrix_multiplication_is_associative_within_tolerance() {
    let mut rng = Rng::new(41);
    let make = |m: usize, n: usize, rng: &mut Rng| {
        Tensor::from_vec((0..m * n).map(|_| rng.range(-1.0, 1.0)).collect(), [m, n]).unwrap()
    };
    let a = make(6, 5, &mut rng);
    let b = make(5, 7, &mut rng);
    let c = make(7, 4, &mut rng);

    let left = a.matmul(&b).unwrap().matmul(&c).unwrap();
    let right = a.matmul(&b.matmul(&c).unwrap()).unwrap();
    assert_close(&left.to_vec(), &right.to_vec(), 1e-12, "associativity");
}

#[test]
fn matmul_matches_naive_triple_loop() {
    let mut rng = Rng::new(53);
    for (m, k, n) in [(1, 1, 1), (3, 4, 5), (9, 7, 2), (16, 16, 16), (33, 17, 8)] {
        let a_values: Vec<f64> = (0..m * k).map(|_| rng.range(-2.0, 2.0)).collect();
        let b_values: Vec<f64> = (0..k * n).map(|_| rng.range(-2.0, 2.0)).collect();
        let a = Tensor::from_vec(a_values.clone(), [m, k]).unwrap();
        let b = Tensor::from_vec(b_values.clone(), [k, n]).unwrap();
        let got = a.matmul(&b).unwrap().to_vec();

        let mut expected = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += a_values[i * k + p] * b_values[p * n + j];
                }
                expected[i * n + j] = acc;
            }
        }
        assert!(
            max_abs_diff(&got, &expected) < 1e-12,
            "matmul ({m},{k})x({k},{n}) max diff {}",
            max_abs_diff(&got, &expected)
        );
    }
}

#[test]
fn elementwise_ops_are_bit_stable_across_layouts() {
    // The vectorised and strided kernels must produce identical bits for
    // exactly-rounded operations.
    let mut rng = Rng::new(61);
    let values: Vec<f64> = (0..64).map(|_| rng.range(-1.0, 1.0)).collect();
    let dense = Tensor::from_vec(values.clone(), [8, 8]).unwrap();
    let transposed = dense.transpose();

    let lhs = dense.mul(&dense).unwrap();
    let rhs = transposed.mul(&transposed).unwrap();
    // Both products contain the same set of squared values, so sorted vectors
    // must agree bit for bit.
    let mut lhs_sorted = lhs.to_vec();
    let mut rhs_sorted = rhs.to_vec();
    lhs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rhs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(lhs_sorted, rhs_sorted);
}

#[test]
fn edge_shapes_are_handled() {
    // 0-d tensors.
    let scalar = Tensor::scalar(2.5);
    assert_eq!(scalar.add(&Tensor::scalar(1.5)).unwrap().scalar_value().unwrap(), 4.0);
    // Empty tensors.
    let empty = Tensor::zeros([0, 4]);
    assert_eq!(empty.add(&Tensor::ones([0, 4])).unwrap().len(), 0);
    assert_eq!(empty.sum(), 0.0);
    // Single element dimensions.
    let singleton = Tensor::from_vec(vec![3.0], [1, 1, 1]).unwrap();
    assert_eq!(singleton.muls(2.0).to_vec(), vec![6.0]);
    // Very large strides from a narrow view.
    let base = Tensor::arange(0.0, 100.0, 1.0).unwrap();
    let strided = base.narrow(0, 90, 10).unwrap();
    assert_eq!(strided.sum(), 945.0);
}

#[test]
fn numerical_edge_cases_follow_ieee() {
    let values = vec![1.0, 0.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
    let tensor = Tensor::from_vec(values, [6]).unwrap();
    let ln = tensor.ln().to_vec();
    assert_eq!(ln[0], 0.0);
    assert_eq!(ln[1], f64::NEG_INFINITY);
    assert!(ln[2].is_nan());
    assert!(ln[3].is_infinite());
    assert!(ln[5].is_nan());
    assert!(tensor.has_nan());
    assert!(!tensor.all_finite());
    assert!(tensor.max().is_nan(), "NaN must propagate through max");
}
