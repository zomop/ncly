//! Matrix norms shared by the factorisations.

/// The `inf`-norm of a row-major `n × n` matrix: `max_i Σ_j |a_ij|`.
///
/// This is the natural scale for relative pivot tests because it is cheap
/// (`O(n²)`), it is a genuine norm (so `κ_∞` follows from it), and it is
/// invariant under row permutation — which matters when the pivot search itself
/// reorders rows.
pub fn matrix_norm_inf(data: &[f64], n: usize) -> f64 {
    let mut best = 0.0f64;
    for row in data.chunks(n.max(1)) {
        let total: f64 = row.iter().map(|v| v.abs()).sum();
        if total > best {
            best = total;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_is_largest_row_sum() {
        let a = [1.0, -2.0, 3.0, 4.0];
        assert_eq!(matrix_norm_inf(&a, 2), 7.0);
        assert_eq!(matrix_norm_inf(&[], 0), 0.0);
    }
}
