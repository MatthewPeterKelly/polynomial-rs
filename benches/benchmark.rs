use std::f64::consts::PI;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use iter_num_tools::lin_space;
use num_bigint::BigInt;
use num_complex::Complex;
use num_rational::BigRational;
use polynomial::Polynomial;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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

fn eval_high_precision_polynomial(_num_samples: usize) {
    let val_1 = BigRational::new(BigInt::from(1), BigInt::from(3));
    let val_2 = BigRational::new(BigInt::from(2), BigInt::from(3));
    let val_3 = BigRational::new(BigInt::from(5), BigInt::from(7));
    let poly_3 = Polynomial::new([val_1, val_2, val_3].to_vec());

    black_box(poly_3.eval(BigRational::new(BigInt::from(3), BigInt::from(4))));
}

fn sample_complex_vector<R, D>(num_samples: usize, rng: &mut R, dist: &D) -> Vec<Complex<f64>>
where
    R: Rng,
    D: Distribution<f64>,
{
    (0..num_samples)
        .map(|_| Complex::new(dist.sample(rng), dist.sample(rng)))
        .collect()
}

fn eval_complex_polynomial(sqrt_num_samples: usize) {
    let mut rng = StdRng::seed_from_u64(12345);
    let dist = Uniform::from(-9.0..9.0);

    let poly_3 = Polynomial::new(sample_complex_vector(3, &mut rng, &dist));
    let poly_4 = Polynomial::new(sample_complex_vector(4, &mut rng, &dist));
    let poly_5 = Polynomial::new(sample_complex_vector(5, &mut rng, &dist));
    let real_iter = lin_space(-5.0..=5.0, sqrt_num_samples);
    for real in real_iter {
        let imag_iter = lin_space(-5.0..=5.0, sqrt_num_samples);
        for imag in imag_iter {
            let value = Complex::new(real, imag);
            black_box(poly_3.eval(value));
            black_box(poly_4.eval(value));
            black_box(poly_5.eval(value));
        }
    }
}

fn benchmark(c: &mut Criterion) {
    let num_samples = 5000;
    c.bench_function("eval_scalar_polynomial", |b| {
        b.iter(|| eval_scalar_polynomial(num_samples))
    });
    c.bench_function("eval_complex_polynomial", |b| {
        b.iter(|| eval_complex_polynomial( (num_samples as f64).sqrt() as usize))
    });
    c.bench_function("eval_high_precision_polynomial", |b| {
        b.iter(|| eval_high_precision_polynomial(num_samples))
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
