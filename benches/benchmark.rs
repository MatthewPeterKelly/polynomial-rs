use std::f64::consts::PI;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use iter_num_tools::lin_space;
use polynomial::Polynomial;

fn eval_scalar_polynomial(num_samples: usize) {
    let poly_quadratic = Polynomial::new(vec![-5.0, 2.5, 1.0]);
    let poly_sin_cubic = Polynomial::chebyshev(&f64::sin, 4, -PI, PI).unwrap();
    let poly_cos_quartic = Polynomial::chebyshev(&f64::cos, 5, -PI, PI).unwrap();
    let queries = lin_space(-5.0..=5.0, num_samples);
    for angle in queries {
        black_box(poly_quadratic.eval(angle));
        black_box(poly_sin_cubic.eval(angle));
        black_box(poly_cos_quartic.eval(angle));
    }
}

fn benchmark(c: &mut Criterion) {
    c.bench_function("eval_cubic_scalar_polynomial", |b| {
        b.iter(|| eval_scalar_polynomial(5000))
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
