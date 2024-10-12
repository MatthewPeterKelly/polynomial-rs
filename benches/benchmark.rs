use std::f64::consts::PI;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use iter_num_tools::lin_space;
use num_complex::Complex;
use polynomial::Polynomial;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rug::Float;

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

fn eval_high_precision_polynomial(num_samples: usize, precision: u32) {
    let mut rng = StdRng::seed_from_u64(12345);
    let dist = Uniform::from(-9.0..9.0);
    let mut sample_vector = |length: usize| -> Vec<Float> {
        (0..length)
            .map(|_| Float::with_value(precision, dist.sample(&mut rng)))
            .collect()
    };
    let poly_3 = Polynomial::new(sample_complex_vector(3));
    let poly_4 = Polynomial::new(sample_complex_vector(4));
    let poly_5 = Polynomial::new(sample_complex_vector(5));
    for angle in queries {
        black_box(poly_3.eval(value));
        black_box(poly_4.eval(value));
        black_box(poly_5.eval(value));
    }
}

fn eval_complex_polynomial(num_samples: usize) {
    let mut rng = StdRng::seed_from_u64(12345);
    let dist = Uniform::from(-9.0..9.0);
    let mut sample_complex_vector = |length: usize| -> Vec<Complex<f64>> {
        (0..length)
            .map(|_| Complex::new(dist.sample(&mut rng), dist.sample(&mut rng)))
            .collect()
    };

    let poly_3 = Polynomial::new(sample_complex_vector(3));
    let poly_4 = Polynomial::new(sample_complex_vector(4));
    let poly_5 = Polynomial::new(sample_complex_vector(5));
    let sqrt_sample_count = (num_samples as f64).sqrt() as usize;
    let real_iter = lin_space(-5.0..=5.0, sqrt_sample_count);
    for real in real_iter {
        let imag_iter = lin_space(-5.0..=5.0, sqrt_sample_count);
        for imag in imag_iter {
            let value = Complex::new(real, imag);
            black_box(poly_3.eval(value));
            black_box(poly_4.eval(value));
            black_box(poly_5.eval(value));
        }
    }
}

fn benchmark(c: &mut Criterion) {
    c.bench_function("eval_scalar_polynomial", |b| {
        b.iter(|| eval_scalar_polynomial(5000))
    });
    c.bench_function("eval_complex_polynomial", |b| {
        b.iter(|| eval_complex_polynomial(5000))
    });
    c.bench_function("eval_high_precision_polynomial", |b| {
        b.iter(|| eval_high_precision_polynomial(5000, 250))
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
