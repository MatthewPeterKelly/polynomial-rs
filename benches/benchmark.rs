use std::f64::consts::PI;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use polynomial::Polynomial;
use iter_num_tools::lin_space;

fn eval_cubic_scalar_polynomial() {
    let poly = Polynomial::chebyshev(&f64::sin, 4, -PI, PI).unwrap();
    let queries = lin_space(-5.0..=5.0, 100);
    for angle in queries {
        let value = poly.eval(angle);
        black_box(value);
    }
}

fn benchmark(c: &mut Criterion) {
    c.bench_function("eval_cubic_scalar_polynomial", |b| b.iter(eval_cubic_scalar_polynomial));
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
