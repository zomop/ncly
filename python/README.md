# NCLY for Python

Python bindings for [NCLY](../README.md), a numerical computing library written in
Rust. The extension is built with [PyO3](https://pyo3.rs) and
[maturin](https://maturin.rs).

```bash
pip install maturin
cd python
maturin develop --release
python -c "import ncly; print(ncly.__version__)"
```

See the main README for the API tour and `tests/` for executable examples.
