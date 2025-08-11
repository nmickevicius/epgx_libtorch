/**
 * @file epgx_mt_gre.cpp
 * @brief Gradient-echo EPG-X simulator for a two-pool magnetization-transfer (MT) system (PyTorch C++ extension).
 *
 * @details
 * This implements an EPG-X (extended phase graph with chemical/exchange) model for a two-pool MT system:
 * free pool "a" (observable) and bound/saturated pool "b" (unobservable). Each TR:
 *   1) Apply an RF pulse (flip α, phase φ) to the transverse/longitudinal states via a 4×4 RF operator T.
 *   2) Sample F0 (demodulated by e^{-iφ}).
 *   3) Apply relaxation/exchange during TR via a block-diagonal propagator Xi and add a recovery vector b
 *      for the longitudinal states (Za, Zb).
 *   4) Apply gradient spoiling: shift F⁺/F⁻ states across EPG orders and enforce F⁺₀ = conj(F⁻₀).
 *
 * State ordering for each EPG order k:
 *   [ F⁺_k, F⁻_k, Z_a,k, Z_b,k ]ᵀ  (complex, size 4)
 *
 * Inputs are per-pulse sequences of scalars (length = num_pulses), dtype float64, on a single device (CPU/CUDA).
 * Outputs are complex<double>.
 *
 * Notation (per-element, length num_pulses unless noted):
 *   α (alpha): flip angle [rad]
 *   φ (phi): RF phase [rad]
 *   WT: effective MT saturation weight during the RF pulse for pool b (see example call)
 *   TR: repetition time per pulse [ms]
 *   T1a, T1b, T2a: relaxation times [ms]
 *   f: pool-b proton fraction (0..1); M0a = 1−f, M0b = f
 *   k_a: exchange rate from a→b [ms⁻¹]; k_b is computed internally as k_b = k_a * M0a / M0b
 *   kmax: number of EPG orders retained (k > 0)
 *
 * Numerics:
 * - All real inputs must be float64 (double). Complex outputs are complex<double>.
 * - expm2x2_real() uses a stable closed form (series near s→0) for a real 2×2 matrix.
 * - This implementation assumes ideal spoiling between TRs (shift-and-conjugate).
 *
 * PyBind exports: epgx_mt_gre(alpha, phi, WT, TR, T1a, T1b, T2a, f, ka, kmax) -> complex tensor (F0 over pulses).
 */

#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <c10/util/complex.h>
#include <tuple>

using c64 = c10::complex<double>;
using torch::indexing::Slice;
using torch::indexing::None;

/** @brief Build the 4×4 complex RF operator for one pulse.
 *
 * The operator acts on [F⁺, F⁻, Z_a, Z_b]ᵀ. It includes rotation of
 * transverse/longitudinal components with flip α and phase φ. The (3,3) element
 * is set to exp(-WT) to model effective saturation of pool b during the pulse.
 *
 * @param a  Scalar flip angle α [rad], float64.
 * @param p  Scalar RF phase φ [rad], float64.
 * @param WT Scalar effective MT saturation weight for pool b (dimensionless), float64.
 * @return   4×4 complex<double> tensor on the same device as inputs.
 *
 * @pre a,p,WT are scalar float64 tensors on the same device.
 * @note No side effects; pure function.
 */
torch::Tensor get_rf_operator(const torch::Tensor& a, const torch::Tensor& p, const torch::Tensor& WT);

/** @brief Build relaxation/exchange propagator over a time step dt (=TR[n]).
 *
 * Splits into transverse (R2a) and longitudinal (R1a/R1b with exchange k_a,k_b) blocks:
 *   - Xi_T = diag(exp(-R2a·dt), exp(-R2a·dt))
 *   - Xi_L = expm( [ [-R1a-k_a,  k_b],
 *                    [  k_a,    -R1b-k_b] ] · dt )
 *
 * Returns a 4×4 complex block-diagonal Xi and a length-4 complex recovery vector b
 * with non-zeros only in Za/Zb slots. b = (Xi_L − I) · Λ_L^{-1} · [M0a R1a, M0b R1b]ᵀ.
 *
 * @param R1a  1/T1a [ms⁻¹], scalar float64.
 * @param R1b  1/T1b [ms⁻¹], scalar float64.
 * @param R2a  1/T2a [ms⁻¹], scalar float64.
 * @param M0a  Equilibrium longitudinal mag. of pool a (=1−f), scalar float64.
 * @param M0b  Equilibrium longitudinal mag. of pool b (=f), scalar float64.
 * @param ka   Exchange rate a→b [ms⁻¹], scalar float64.
 * @param kb   Exchange rate b→a [ms⁻¹], scalar float64.
 * @param dt   Time step [ms], scalar float64 (typically TR).
 * @return     (Xi [4×4 complex], b [4 complex]).
 */
std::tuple<torch::Tensor, torch::Tensor> get_relaxation_exchange_operator(
    const torch::Tensor& R1a, const torch::Tensor& R1b, const torch::Tensor& R2a,
    const torch::Tensor& M0a, const torch::Tensor& M0b, const torch::Tensor& ka,
    const torch::Tensor& kb, const torch::Tensor& dt);

/** @brief Real 2×2 matrix exponential via closed form (stable near s→0).
 *
 * Computes exp(A) using the decomposition A = (tr/2)I + B with tr = trace(A).
 * Then exp(A) = e^{tr/2} [ cosh(s)·I + (sinh(s)/s)·B ], where s² = (tr/2)² − det(A).
 * Uses a series expansion for sinh(s)/s when |s| is small.
 *
 * @param A [2×2], float64, CPU or CUDA.
 * @return  [2×2], float64, on A.device().
 */
torch::Tensor expm2x2_real(const torch::Tensor& A);

/** @brief Solve a 2×2 linear system A·x = b using an explicit inverse.
 *
 * @param A [2×2], float/double.
 * @param b [2],   float/double.
 * @return  x [2], same dtype/device as inputs.
 * @pre det(A) ≠ 0.
 */
torch::Tensor solve2x2(const torch::Tensor& A, const torch::Tensor& b);

/** @brief Gradient-echo EPG-X simulator (two-pool MT) producing the F0 signal train.
 *
 * For each pulse n:
 *   1) Build RF operator T(α[n], φ[n], WT[n]) and apply: Ω ← T·Ω.
 *   2) Sample F0[n] = F⁺₀ · e^{-iφ[n]} (demodulated to the rotating frame).
 *   3) Build (Xi, b) from relaxation/exchange using TR[n] and apply: Ω ← Xi·Ω, then
 *      add b to Za₀ and Zb₀ (longitudinal recovery at k=0 only).
 *   4) Spoil: shift F⁺ up in k, F⁻ down in k, enforce F⁺₀ = conj(F⁻₀).
 *
 * @param alpha  [num_pulses] flip angles [rad], float64.
 * @param phi    [num_pulses] RF phases [rad], float64.
 * @param WT     [num_pulses] effective MT saturation weight for pool b, float64.
 * @param TR     [num_pulses] repetition times [ms], float64.
 * @param T1a    [num_pulses] T1 of pool a [ms], float64.
 * @param T1b    [num_pulses] T1 of pool b [ms], float64.
 * @param T2a    [num_pulses] T2 of pool a [ms], float64.
 * @param f      [num_pulses] pool-b proton fraction (0..1), float64.
 * @param ka     [num_pulses] exchange rate a→b [ms⁻¹], float64.
 * @param kmax   number of retained EPG orders (k ≥ 1).
 * @return       Complex<double> tensor [num_pulses] with the demodulated F0 per TR on input device.
 *
 * @pre All 1-D inputs have the same length and dtype float64 on the same device.
 * @pre kmax ≥ 1.
 * @note kb is computed internally assuming detailed balance: kb = ka * M0a / M0b.
 */
torch::Tensor epgx_mt_gre(
    torch::Tensor alpha,
    torch::Tensor phi,
    torch::Tensor WT,
    torch::Tensor TR,
    torch::Tensor T1a,
    torch::Tensor T1b,
    torch::Tensor T2a,
    torch::Tensor f,
    torch::Tensor ka,
    int kmax
) {

    // ----- Dimensions / derived rates -----
    auto num_pulses = alpha.numel();

    auto R1a = 1.0 / T1a;
    auto R1b = 1.0 / T1b;
    auto R2a = 1.0 / T2a;
    auto M0a = 1.0 - f;
    auto M0b = f;
    auto kb = ka * M0a / M0b;  // detailed balance

    // ----- Outputs / state allocation -----
    auto F0 = torch::zeros({num_pulses}, torch::dtype(torch::kComplexDouble).device(R1a.device()));

    // Ω: [4 × kmax] complex state matrix across EPG orders
    auto Omega = torch::zeros({4, kmax}, torch::dtype(torch::kComplexDouble).device(R1a.device()));
    Omega.index_put_({2,0}, 1.0 - f.index({0}));  // Za,0 = M0a at start
    Omega.index_put_({3,0}, f.index({0}));        // Zb,0 = M0b at start

    // ----- Main loop over pulses -----
    for (int64_t n = 0; n < num_pulses; n++) {

        // (1) RF operator for this pulse: T ∈ ℂ^{4×4}
        auto T = get_rf_operator(alpha.select(0,n), phi.select(0,n), WT.select(0,n));

        // Apply RF: Ω ← T·Ω
        Omega = torch::matmul(T, Omega);

        // (2) Sample demodulated F0 (use -iφ so signal is brought to rotating frame)
        F0.index_put_({n}, Omega.index({0,0}) * torch::exp(-1.0 * phi.index({n}).to(torch::kComplexDouble) * c64(0.0, 1.0)));

        // (3) Relaxation/exchange over TR[n] at k=0 for Za/Zb, and transverse decay
        auto [Xi, b] = get_relaxation_exchange_operator(
            R1a.select(0,n), R1b.select(0,n), R2a.select(0,n),
            M0a.select(0,n), M0b.select(0,n), ka.select(0,n), kb.select(0,n),
            TR.select(0,n));

        // Apply propagator
        Omega = torch::matmul(Xi, Omega);

        // Add longitudinal recovery at k=0: Ω_z,0 ← Ω_z,0 + b
        for (int z = 2; z < 4; z++)
            Omega.index_put_({z,0}, Omega.index({z,0}) + b.index({z}));

        // (4) Gradient spoiling: shift F⁺ up and F⁻ down in k; enforce F⁺₀ = conj(F⁻₀)
        auto F_src  = Omega.index({0, Slice(None, -1)}).clone();
        Omega.index_put_({0, Slice(1, None)}, F_src);   // shift F⁺ states up
        auto Fm_src = Omega.index({1, Slice(1, None)}).clone();
        Omega.index_put_({1, Slice(None, -1)}, Fm_src); // shift F⁻ states down
        Omega.index_put_({0, 0}, torch::conj(Omega.index({1, 0})));
    }

    return F0;
}

// ========================= Implementation helpers =========================

torch::Tensor get_rf_operator(
    const torch::Tensor& a,
    const torch::Tensor& p,
    const torch::Tensor& WT
) {
  // Expect: a, p, WT are scalar double tensors (CPU or CUDA)
  TORCH_CHECK(a.scalar_type() == torch::kFloat64, "a must be float64 (double)");
  TORCH_CHECK(p.scalar_type() == torch::kFloat64, "p must be float64 (double)");
  TORCH_CHECK(WT.scalar_type() == torch::kFloat64, "WT must be float64 (double)");
  TORCH_CHECK(a.numel() == 1 && p.numel() == 1 && WT.numel() == 1, "a, p, WT must be scalars");

  auto dev = a.device();
  constexpr auto CD = torch::kComplexDouble;
  constexpr auto RD = torch::kFloat64;

  // Precompute trig/exponentials
  auto half = a / 2.0;
  auto c = torch::cos(half);         // cos(a/2)   (real)
  auto s = torch::sin(half);         // sin(a/2)   (real)
  auto sa = torch::sin(a);           // sin(a)     (real)
  auto ca = torch::cos(a);           // cos(a)     (real)

  auto pC = p.to(CD); // complex view of p for exp(i·p)
  auto e_minus_ip  = torch::exp(pC * c64(0.0, -1.0));
  auto e_plus_ip   = torch::exp(pC * c64(0.0,  1.0));
  auto e_minus_2ip = torch::exp(pC * c64(0.0, -2.0));

  // 4×4 RF operator in state ordering [F⁺, F⁻, Z_a, Z_b]
  auto T = torch::zeros({4, 4}, torch::dtype(CD).device(dev));

  T.index_put_({0, 0}, (c * c).to(CD));                         // Tap(1)  = cos(a/2)^2
  T.index_put_({1, 0}, e_minus_2ip * (s * s).to(CD));           // Tap(2)  = exp(-2i p) * sin(a/2)^2
  T.index_put_({2, 0}, e_minus_ip * sa.to(CD) * c64(0.0,-0.5)); // Tap(3)  = -0.5i * exp(-i p) * sin(a)

  T.index_put_({0, 1}, torch::conj(T.index({1, 0})) );          // Tap(5)  = conj(Tap(2))
  T.index_put_({1, 1}, T.index({0, 0}) );                       // Tap(6)  = Tap(1)
  T.index_put_({2, 1}, e_plus_ip * sa.to(CD) * c64(0.0, 0.5));  // Tap(7)  = 0.5i * exp(i p) * sin(a)

  T.index_put_({0, 2}, e_plus_ip * sa.to(CD) * c64(0.0,-1.0));  // Tap(9)  = -i * exp(i p) * sin(a)
  T.index_put_({1, 2}, e_minus_ip * sa.to(CD) * c64(0.0, 1.0)); // Tap(10) =  i * exp(-i p) * sin(a)
  T.index_put_({2, 2}, ca.to(CD));                              // Tap(11) = cos(a)

  T.index_put_({3, 3}, torch::exp(-WT).to(CD));                 // Tap(16) = exp(-WT)  (MT saturation of pool b)

  return T;
}

std::tuple<torch::Tensor, torch::Tensor> get_relaxation_exchange_operator(
    const torch::Tensor& R1a, const torch::Tensor& R1b, const torch::Tensor& R2a,
    const torch::Tensor& M0a, const torch::Tensor& M0b, const torch::Tensor& ka,
    const torch::Tensor& kb, const torch::Tensor& dt
) {
    auto dev = R2a.device();
    constexpr auto RD = torch::kFloat64;
    constexpr auto CD = torch::kComplexDouble;

    // Transverse decay for pool a (F⁺/F⁻): Xi_T = diag(e^{-R2a·dt})
    auto Xi_T = torch::zeros({2,2}, torch::dtype(RD).device(dev));
    Xi_T.index_put_({0,0}, torch::exp(-R2a * dt));
    Xi_T.index_put_({1,1}, torch::exp(-R2a * dt));

    // Longitudinal 2×2 with exchange
    auto Lambda_L = torch::zeros({2,2}, torch::dtype(RD).device(dev));
    Lambda_L.index_put_({0,0}, -R1a - ka);
    Lambda_L.index_put_({0,1},  kb);
    Lambda_L.index_put_({1,0},  ka);
    Lambda_L.index_put_({1,1}, -R1b - kb);
    auto Xi_L = expm2x2_real(Lambda_L * dt);  // real 2×2

    // Composite Xi: block-diag{Xi_T, Xi_L} in complex space
    auto Xi = torch::zeros({4,4}, torch::dtype(CD).device(dev));
    Xi.index_put_({0,0}, Xi_T.index({0,0}).to(CD));
    Xi.index_put_({1,1}, Xi_T.index({1,1}).to(CD));
    Xi.index_put_({2,2}, Xi_L.index({0,0}).to(CD));
    Xi.index_put_({2,3}, Xi_L.index({0,1}).to(CD));
    Xi.index_put_({3,2}, Xi_L.index({1,0}).to(CD));
    Xi.index_put_({3,3}, Xi_L.index({1,1}).to(CD));

    // Recovery vector: b has non-zero entries only for Za/Zb
    auto C_L = torch::zeros({2}, torch::dtype(RD).device(dev));
    C_L.index_put_({0}, M0a * R1a);
    C_L.index_put_({1}, M0b * R1b);

    auto I2 = torch::eye(2, torch::dtype(RD).device(dev));
    auto Xi_L_minus_I = Xi_L - I2;

    auto Lambda_L_inv_C_L = solve2x2(Lambda_L, C_L);          // [2]
    auto Z_off = torch::matmul(Xi_L_minus_I, Lambda_L_inv_C_L);// [2]

    auto b = torch::zeros({4}, torch::dtype(CD).device(dev));
    b.index_put_({2}, Z_off.index({0}).to(CD));
    b.index_put_({3}, Z_off.index({1}).to(CD));

    return std::make_tuple(Xi, b);
}

torch::Tensor expm2x2_real(const torch::Tensor& A) {
  // A: [2,2], float64, CPU or CUDA
  auto opts = A.options();
  auto a = A.index({0,0});
  auto b = A.index({0,1});
  auto c = A.index({1,0});
  auto d = A.index({1,1});

  auto tr  = a + d;                 // trace
  auto det = a*d - b*c;             // determinant
  auto s2  = tr*tr/4.0 - det;       // (trace/2)^2 - det
  auto s   = torch::sqrt(torch::clamp(s2, 0.0)); // real branch

  auto I   = torch::eye(2, opts);
  auto e   = torch::exp(tr/2.0);

  // B = A - (tr/2) I
  auto B = A.clone();
  B.index_put_({0,0}, a - tr/2.0);
  B.index_put_({1,1}, d - tr/2.0);

  // Stable near s ≈ 0: sinh(s)/s ≈ 1 + s^2/6
  auto eps = 1e-12;
  auto sinh_s_over_s = torch::where(torch::abs(s) < eps,
                                    1.0 + s*s/6.0,
                                    torch::sinh(s)/s);

  auto E = e * (torch::cosh(s) * I + sinh_s_over_s * B);
  return E;
}

torch::Tensor solve2x2(const torch::Tensor& A, const torch::Tensor& b) {
    // A: [2,2] tensor (float or double)
    // b: [2]   tensor
    TORCH_CHECK(A.size(0) == 2 && A.size(1) == 2, "A must be 2x2");
    TORCH_CHECK(b.size(0) == 2, "b must be length 2");

    auto a = A.index({0,0});
    auto bb = A.index({0,1});  // avoid clash with vector b
    auto c = A.index({1,0});
    auto d = A.index({1,1});
    auto det = a * d - bb * c;

    auto inv00 =  d / det;
    auto inv01 = -bb / det;
    auto inv10 = -c / det;
    auto inv11 =  a / det;

    auto x0 = inv00 * b.index({0}) + inv01 * b.index({1});
    auto x1 = inv10 * b.index({0}) + inv11 * b.index({1});

    return torch::stack({x0, x1});
}

// ========================= PyTorch binding =========================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "epgx_mt_gre",
        &epgx_mt_gre,
        R"doc(
gradient-echo EPG-X simulator for two-pool magnetization-transfer systems.

Args:
  alpha (Tensor[float64], shape [N]): flip angles in radians.
  phi   (Tensor[float64], shape [N]): RF phases in radians.
  WT    (Tensor[float64], shape [N]): effective MT saturation weight applied to pool b during RF.
  TR    (Tensor[float64], shape [N]): repetition times (ms)
  T1a   (Tensor[float64], shape [N]): T1 of pool a (ms).
  T1b   (Tensor[float64], shape [N]): T1 of pool b (ms).
  T2a   (Tensor[float64], shape [N]): T2 of pool a (ms).
  f     (Tensor[float64], shape [N]): pool-b proton fraction in [0,1]; M0a=1-f, M0b=f.
  ka    (Tensor[float64], shape [N]): exchange rate a→b (ms⁻¹); kb is computed as ka*M0a/M0b.
  kmax  (int): number of EPG orders (≥1).

Returns:
  Tensor[complex128], shape [N]: demodulated F0 signal at each TR on the input device.

Notes:
  - All real inputs must be float64 on the same device (CPU or CUDA). Output is complex128.
  - Time units just need to be consistent (e.g., ms everywhere). The example below uses milliseconds.
  - Ideal spoiling model: F⁺ shifts up in k, F⁻ down in k, and F⁺₀ = conj(F⁻₀).

Example (Python)
----------------
```python
import torch
import numpy as np
from epgx_mt import epgx_mt_gre
dtype = torch.float64

npulses = 100
kmax = 50
trf = 0.5  # ms
G = 14e-3  # ms
gam = 267.5221 * 1e-3  # rad/ms/uT

alpha = (10*np.pi/180) * torch.ones((npulses,), dtype=dtype)  # rad
p = torch.arange(npulses, dtype=dtype) + 1.0
phi = p*(p-1)/2 * (117 * np.pi / 180)  # rad
WT = np.pi * gam**2 * (alpha/(gam * trf))**2 * trf * G
TR = 5 * torch.ones((npulses,), dtype=dtype)    # ms
T1a = 750 * torch.ones((npulses,), dtype=dtype) # ms
T1b = 750 * torch.ones((npulses,), dtype=dtype) # ms
T2a = 70 * torch.ones((npulses,), dtype=dtype)  # ms
f = 0.1 * torch.ones((npulses,), dtype=dtype)   # fraction < 1
ka = 4.3e-3 * torch.ones((npulses,), dtype=dtype)  # 1/ms

signal = epgx_mt_gre(alpha, phi, WT, TR, T1a, T1b, T2a, f, ka, kmax)
)doc"
    );
}
