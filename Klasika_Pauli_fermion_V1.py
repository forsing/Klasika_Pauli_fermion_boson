#!/usr/bin/env python3



"""
V1 FERMION — Loto 7/39
Python 3.11
Dependencies: numpy, scipy

Conditional Slater distribution:

    A(x) = diag(exp(h(x))) B
    h(x) = CAP * tanh((x W) / CAP)

    P(S | x) = |det(A(x)[S, :])|^2 / det(A(x)^H A(x))

B is a learned complex matrix with 39 rows and 7 columns.
W is learned by distribution regression.

This program computes fermionic amplitudes classically.
It does not execute a quantum circuit.

No random initialization, sampling, row shuffling or random splits.
No classifier.
No rounding or duplicate repair.
No combination filters.

Chronological validation:
    Train first 60%; validate next 10%.
    Train first 70%; validate next 10%.
    Select parameters using validation NLL only.
    Train first 80%; evaluate final 20% once.
    Fit the selected model on ALL historical transitions.

Test inputs use only draws preceding their targets.
Test targets never enter parameter fitting or feature fitting.

NEXT:
    Exhaustive search over all legal seven-number combinations.
    Sum all model probabilities during that search.
    Report the numerical maximum of the fitted distribution.

Numerical correction:
    Handle floating-point flags locally inside checked_slogdet.
    Validate every returned phase and logarithm.
    Allow zero determinants where zero amplitudes are valid.
    Reject singular normalization matrices.
    Keep global numerical controls enabled elsewhere.
"""



import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import csv
import hashlib
import math
import sys
import traceback
from dataclasses import dataclass
from itertools import combinations, islice
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp



FILES = (
    (
        Path("/data/loto7_4710_k86.csv"),
        4710,
    ),
    (
        Path("/data/loto7_4710_k86_loto_2977.csv"),
        2977,
    ),
    (
        Path("/data/loto7_4710_k86_loto_plus_1733.csv"),
        1733,
    ),
)



NUMBER_COUNT = 39
DRAW_SIZE = 7

FEATURE_DIM = 8
CAP = 4.0
GAUGE_WEIGHT = 0.1

CANDIDATES = (
    (1, 0.03),
    (1, 0.30),
    (3, 0.03),
    (3, 0.30),
)

INITIAL_ORDERS = (17, 11)

OPTIMIZER_STEPS = 600
OPTIMIZER_RESTARTS = 3
GRADIENT_LIMIT = 5e-4

MAP_BATCH = 4096
NORMALIZATION_TOLERANCE = 2e-7


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checked_slogdet(matrix):
    """
    Compute slogdet with local floating-point flag handling.

    A zero determinant must return:
        sign = 0
        logarithm = -inf

    A nonzero determinant must return:
        a finite unit-modulus sign
        a finite logarithm

    Numerical flags from the backend are handled only inside this
    call. The returned values are always checked.
    """
    matrix = np.asarray(matrix)

    require(
        matrix.ndim >= 2
        and matrix.shape[-1] == matrix.shape[-2]
        and matrix.shape[-1] > 0,
        "Determinanta zahteva nepraznu kvadratnu matricu.",
    )
    require(
        np.all(np.isfinite(matrix)),
        "Ulaz determinante sadrži NaN ili beskonačnost.",
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
        under="ignore",
    ):
        sign, logarithm = np.linalg.slogdet(matrix)

    require(
        np.all(np.isfinite(sign)),
        "Neispravna faza determinante.",
    )

    zero = sign == 0

    valid_zero = zero & np.isneginf(logarithm)
    valid_nonzero = (
        (~zero)
        & np.isfinite(logarithm)
        & (np.abs(np.abs(sign) - 1.0) < 1e-7)
    )

    require(
        np.all(valid_zero | valid_nonzero),
        "Neispravan rezultat determinante.",
    )

    return sign, logarithm


def positive_logdet(gram):
    sign, logarithm = checked_slogdet(gram)

    require(
        np.all(np.isfinite(logarithm)),
        "Singularna ili neispravna normalizaciona matrica.",
    )
    require(
        np.all(np.abs(sign - 1.0) < 1e-6),
        "Normalizaciona matrica nije numerički pozitivno definitna.",
    )

    return logarithm


def load_draws(path, expected_rows):
    raw = path.read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()

    rows = []
    blank_rows = 0

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for line_number, row in enumerate(
            csv.reader(handle),
            start=1,
        ):
            if not row or all(not item.strip() for item in row):
                blank_rows += 1
                continue

            require(
                len(row) == DRAW_SIZE,
                f"{path.name}, red {line_number}: "
                "potrebno je tačno 7 kolona.",
            )

            try:
                numbers = [int(item.strip()) for item in row]
            except ValueError as error:
                raise RuntimeError(
                    f"{path.name}, red {line_number}: "
                    "neispravan ceo broj."
                ) from error

            require(
                all(
                    1 <= value <= NUMBER_COUNT
                    for value in numbers
                ),
                f"{path.name}, red {line_number}: "
                "broj van opsega 1–39.",
            )
            require(
                len(set(numbers)) == DRAW_SIZE,
                f"{path.name}, red {line_number}: "
                "ponovljen broj.",
            )

            # Sort only within a draw.
            # Keep the chronological order of all draws.
            rows.append(
                sorted(value - 1 for value in numbers)
            )

    require(
        len(rows) == expected_rows,
        f"{path.name}: očekivano {expected_rows}, "
        f"pročitano {len(rows)} redova.",
    )

    draws = np.asarray(rows, dtype=np.int64)

    require(
        draws.shape == (expected_rows, DRAW_SIZE),
        "Pogrešan oblik CSV-a.",
    )

    print(
        f"\nULAZ: {path.name}\n"
        f"Redova: {len(draws)} | praznih redova: {blank_rows}\n"
        f"SHA256: {fingerprint}\n"
        "Redosled kola ostaje redosled iz CSV-a.",
        flush=True,
    )

    return draws, fingerprint


def make_context(draws, target_indices, lookback):
    """
    For target t, use only draws t-1 through t-lookback.

    Each lag contains:
        39 occupancy entries;
        1 availability indicator.

    Missing early lags contain zeros and availability=0.
    Every transition from the second draw onward is retained.
    """
    target_indices = np.asarray(
        target_indices,
        dtype=np.int64,
    )

    require(
        target_indices.ndim == 1,
        "Neispravni indeksi ciljeva.",
    )
    require(
        np.all(
            (target_indices >= 1)
            & (target_indices <= len(draws))
        ),
        "Ciljni indeks van opsega.",
    )

    block_size = NUMBER_COUNT + 1

    result = np.zeros(
        (len(target_indices), lookback * block_size),
        dtype=np.float64,
    )

    for lag in range(1, lookback + 1):
        previous = target_indices - lag
        valid = np.flatnonzero(previous >= 0)
        offset = (lag - 1) * block_size

        if valid.size:
            result[
                valid[:, None],
                offset + draws[previous[valid]],
            ] = 1.0

            result[
                valid,
                offset + NUMBER_COUNT,
            ] = 1.0

    return result


def fit_feature_map(raw_x):
    """
    Fit PCA and whitening on training inputs only.

    Whitening scales the features.
    It does not assume a Gaussian distribution of draws.
    """
    mean = raw_x.mean(axis=0)
    centered = raw_x - mean

    covariance = (
        centered.T @ centered / len(centered)
    )

    values, vectors = np.linalg.eigh(covariance)

    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]

    threshold = max(float(values[0]), 1.0) * 1e-12
    available = int(
        np.count_nonzero(values > threshold)
    )
    dimension = min(FEATURE_DIM, available)

    require(
        dimension > 0,
        "Istorijski ulazi nemaju promenljive karakteristike.",
    )

    values = values[:dimension]
    vectors = vectors[:, :dimension].copy()

    # Resolve eigenvector signs deterministically.
    for column in range(dimension):
        pivot = int(
            np.argmax(np.abs(vectors[:, column]))
        )
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1.0

    projection = (
        vectors / np.sqrt(values)[None, :]
    )

    return mean, projection


def pack_parameters(b, w):
    return np.concatenate(
        (
            b.real.ravel(),
            b.imag.ravel(),
            w.ravel(),
        )
    )


def unpack_parameters(
    parameters,
    number_count,
    draw_size,
    feature_count,
):
    size = number_count * draw_size

    b = (
        parameters[:size].reshape(
            number_count,
            draw_size,
        )
        + 1j
        * parameters[size:2 * size].reshape(
            number_count,
            draw_size,
        )
    )

    w = parameters[2 * size:].reshape(
        feature_count,
        number_count,
    )

    return b, w


def log_amplitudes(x, w):
    z = x @ w
    t = np.tanh(z / CAP)

    return CAP * t, 1.0 - t * t


def distribution_terms(b, w, x, targets):
    h, link_derivative = log_amplitudes(x, w)
    weights = np.exp(2.0 * h)

    gram = np.einsum(
        "ni,ia,ib->nab",
        weights,
        b.conj(),
        b,
        optimize=True,
    )

    log_denominator = positive_logdet(gram)

    selected = b[targets]
    _, log_abs_determinant = checked_slogdet(selected)

    # A zero probability for an observed training/validation target
    # gives infinite NLL. Do not replace it with an invented floor.
    require(
        np.all(np.isfinite(log_abs_determinant)),
        "Nulta amplituda posmatranog kola; "
        "računanje gubitka je zaustavljeno.",
    )

    row_indices = (
        np.arange(len(targets))[:, None]
    )

    log_probability = (
        2.0 * log_abs_determinant
        + 2.0
        * np.sum(
            h[row_indices, targets],
            axis=1,
        )
        - log_denominator
    )

    require(
        np.all(np.isfinite(log_probability)),
        "Neispravna log-verovatnoća.",
    )
    require(
        float(np.max(log_probability)) <= 1e-7,
        "Izračunata verovatnoća prelazi 1.",
    )

    return (
        log_probability,
        weights,
        gram,
        selected,
        link_derivative,
    )


def objective(
    parameters,
    x,
    targets,
    ridge,
    number_count,
    draw_size,
):
    b, w = unpack_parameters(
        parameters,
        number_count,
        draw_size,
        x.shape[1],
    )

    (
        log_probability,
        weights,
        gram,
        selected,
        link_derivative,
    ) = distribution_terms(
        b,
        w,
        x,
        targets,
    )

    sample_count = len(targets)

    inverse_gram = np.linalg.inv(gram)
    inverse_selected = np.linalg.inv(selected)

    b_inverse_gram = np.einsum(
        "ia,nab->nib",
        b,
        inverse_gram,
        optimize=True,
    )

    occupation = weights * np.einsum(
        "nib,ib->ni",
        b_inverse_gram,
        b.conj(),
        optimize=True,
    ).real

    gradient_h = 2.0 * occupation
    row_indices = (
        np.arange(sample_count)[:, None]
    )
    gradient_h[row_indices, targets] -= 2.0

    gradient_w = (
        x.T @ (gradient_h * link_derivative)
        / sample_count
        + 2.0 * ridge * w
    )

    gradient_b = (
        2.0
        * np.einsum(
            "ni,nib->ib",
            weights,
            b_inverse_gram,
            optimize=True,
        )
        / sample_count
    )

    selected_gradient = (
        -2.0
        * inverse_selected.conj().transpose(0, 2, 1)
        / sample_count
    )

    np.add.at(
        gradient_b,
        targets.ravel(),
        selected_gradient.reshape(
            -1,
            draw_size,
        ),
    )

    # Fix the redundant column basis.
    # Orthonormalization does not change the represented distribution.
    gauge_error = (
        b.conj().T @ b - np.eye(draw_size)
    )

    gauge_penalty = (
        GAUGE_WEIGHT
        * np.vdot(
            gauge_error,
            gauge_error,
        ).real
    )

    gradient_b += (
        4.0
        * GAUGE_WEIGHT
        * b
        @ gauge_error
    )

    data_nll = -float(
        np.mean(log_probability)
    )

    loss = (
        data_nll
        + ridge * float(np.sum(w * w))
        + float(gauge_penalty)
    )

    gradient = pack_parameters(
        gradient_b,
        gradient_w,
    )

    require(
        np.isfinite(loss),
        "Funkcija gubitka nije konačna.",
    )
    require(
        np.all(np.isfinite(gradient)),
        "Izvod funkcije gubitka nije konačan.",
    )

    return loss, gradient


def initial_basis(
    number_count,
    draw_size,
    phase_order,
):
    require(
        math.gcd(
            phase_order,
            number_count,
        ) == 1,
        "Neispravan deterministički redosled faza.",
    )

    labels = (
        phase_order
        * np.arange(
            number_count,
            dtype=np.int64,
        )
        + 11
    ) % number_count

    angles = (
        2.0
        * np.pi
        * labels[:, None]
        * np.arange(draw_size)[None, :]
        / number_count
    )

    # Deterministic initial amplitudes.
    # These amplitudes are subsequently learned.
    return (
        np.exp(1j * angles)
        / np.sqrt(number_count)
    )


def orthonormal_basis(amplitudes):
    singular_values = np.linalg.svd(
        amplitudes,
        compute_uv=False,
    )

    require(
        singular_values[-1]
        > 1e-12 * singular_values[0],
        "Amplitudna matrica nema pouzdan numerički rang.",
    )

    q, _ = np.linalg.qr(
        amplitudes,
        mode="reduced",
    )

    error = float(
        np.max(
            np.abs(
                q.conj().T @ q
                - np.eye(q.shape[1])
            )
        )
    )

    require(
        error < 1e-9,
        "Neuspešna ortonormalizacija amplituda.",
    )

    return q


@dataclass
class Model:
    b: np.ndarray
    w: np.ndarray
    mean: np.ndarray
    projection: np.ndarray
    lookback: int
    ridge: float
    train_loss: float
    gradient_norm: float
    initial_order: int

    def transform(self, raw_x):
        return (
            raw_x - self.mean
        ) @ self.projection

    def basis(self, raw_row):
        x = self.transform(
            np.asarray(raw_row).reshape(1, -1)
        )

        h, _ = log_amplitudes(
            x,
            self.w,
        )

        amplitudes = (
            np.exp(h[0])[:, None]
            * self.b
        )

        return orthonormal_basis(amplitudes)


def fit_model(
    raw_x,
    targets,
    lookback,
    ridge,
    label,
):
    mean, projection = fit_feature_map(raw_x)
    x = (raw_x - mean) @ projection

    fitted = []

    for phase_order in INITIAL_ORDERS:
        print(
            f"{label}: lag={lookback}, "
            f"lambda={ridge}, "
            f"početne faze={phase_order}, "
            f"trening={len(targets)}",
            flush=True,
        )

        b0 = initial_basis(
            NUMBER_COUNT,
            DRAW_SIZE,
            phase_order,
        )

        w0 = np.zeros(
            (x.shape[1], NUMBER_COUNT),
            dtype=np.float64,
        )

        parameters = pack_parameters(
            b0,
            w0,
        )

        iteration = 0

        def callback(current):
            nonlocal iteration
            iteration += 1

            if iteration % 100 == 0:
                print(
                    f"  Optimizacija: "
                    f"{iteration} iteracija",
                    flush=True,
                )

        def function(current):
            return objective(
                current,
                x,
                targets,
                ridge,
                NUMBER_COUNT,
                DRAW_SIZE,
            )

        result = None
        last_gradient = math.inf
        converged = False

        try:
            for _ in range(OPTIMIZER_RESTARTS):
                result = minimize(
                    function,
                    parameters,
                    method="L-BFGS-B",
                    jac=True,
                    callback=callback,
                    options={
                        "maxiter": OPTIMIZER_STEPS,
                        "maxls": 60,
                        "maxcor": 20,
                        "ftol": 1e-13,
                        "gtol": 1e-6,
                    },
                )

                parameters = result.x.copy()

                _, gradient = function(parameters)
                last_gradient = float(
                    np.max(np.abs(gradient))
                )

                if (
                    result.success
                    and last_gradient <= GRADIENT_LIMIT
                ):
                    converged = True
                    break

            if not converged:
                message = (
                    result.message
                    if result is not None
                    else "bez rezultata"
                )

                print(
                    f"  Početak nije prihvaćen: "
                    f"{message}; "
                    f"max|grad|={last_gradient:.3e}",
                    flush=True,
                )
                continue

            b, w = unpack_parameters(
                parameters,
                NUMBER_COUNT,
                DRAW_SIZE,
                x.shape[1],
            )

            # Right multiplication by an invertible matrix
            # leaves the normalized Slater distribution unchanged.
            b = orthonormal_basis(b)

            normalized_parameters = pack_parameters(
                b,
                w,
            )

            train_loss, _ = function(
                normalized_parameters
            )

            fitted.append(
                Model(
                    b=b.copy(),
                    w=w.copy(),
                    mean=mean.copy(),
                    projection=projection.copy(),
                    lookback=lookback,
                    ridge=ridge,
                    train_loss=float(train_loss),
                    gradient_norm=last_gradient,
                    initial_order=phase_order,
                )
            )

            print(
                "  Prihvaćen lokalni minimum: "
                f"gubitak={train_loss:.8f}, "
                f"max|grad|={last_gradient:.3e}",
                flush=True,
            )

        except (
            RuntimeError,
            np.linalg.LinAlgError,
            FloatingPointError,
        ) as error:
            print(
                f"  Početak nije prihvaćen: {error}",
                flush=True,
            )

    require(
        bool(fitted),
        f"{label}: nijedan deterministički početak "
        "nije prošao numeričke uslove. "
        "NEXT nije izračunat.",
    )

    # Choose initialization by training objective only.
    return min(
        fitted,
        key=lambda model: model.train_loss,
    )


def evaluate_nll(
    model,
    raw_x,
    targets,
):
    total = 0.0

    for start in range(
        0,
        len(targets),
        512,
    ):
        stop = min(
            start + 512,
            len(targets),
        )

        x = model.transform(
            raw_x[start:stop]
        )

        log_probability = distribution_terms(
            model.b,
            model.w,
            x,
            targets[start:stop],
        )[0]

        total -= float(
            np.sum(log_probability)
        )

    return total / len(targets)


def exhaustive_maximum(
    q,
    verbose=True,
    label="NEXT",
):
    number_count, draw_size = q.shape

    expected = math.comb(
        number_count,
        draw_size,
    )

    iterator = combinations(
        range(number_count),
        draw_size,
    )

    visited = 0
    best_log_probability = -math.inf
    best_combination = None
    log_total_probability = -math.inf
    previous_progress = 0

    while True:
        block = list(
            islice(iterator, MAP_BATCH)
        )

        if not block:
            break

        indices = np.asarray(
            block,
            dtype=np.int64,
        )

        selected = q[indices]

        _, log_abs_determinant = checked_slogdet(
            selected
        )

        # Zero determinants give -inf scores.
        # They are valid zero-probability combinations.
        scores = 2.0 * log_abs_determinant

        require(
            not np.any(np.isnan(scores)),
            "NaN tokom završne pretrage.",
        )

        local_index = int(
            np.argmax(scores)
        )
        local_score = float(
            scores[local_index]
        )

        # Enumeration is lexicographic.
        # Exact numerical ties retain the first combination.
        if local_score > best_log_probability:
            best_log_probability = local_score
            best_combination = tuple(
                int(value)
                for value in indices[local_index]
            )

        block_log_total = float(
            logsumexp(scores)
        )

        log_total_probability = float(
            np.logaddexp(
                log_total_probability,
                block_log_total,
            )
        )

        visited += len(indices)

        progress = visited // 2_000_000

        if (
            verbose
            and progress > previous_progress
        ):
            print(
                f"{label}: pregledano "
                f"{visited:,}/{expected:,}",
                flush=True,
            )
            previous_progress = progress

    require(
        visited == expected,
        "Završna pretraga nije potpuna.",
    )
    require(
        best_combination is not None,
        "Nije pronađena kombinacija.",
    )
    require(
        np.isfinite(best_log_probability),
        "Sve kombinacije imaju nultu "
        "numeričku amplitudu.",
    )
    require(
        best_log_probability <= 1e-7,
        "Maksimalna modelna verovatnoća prelazi 1.",
    )

    probability_sum = math.exp(
        log_total_probability
    )

    require(
        abs(probability_sum - 1.0)
        <= NORMALIZATION_TOLERANCE,
        "Normalizacija nije prošla: "
        f"zbir={probability_sum:.12g}.",
    )

    return (
        best_combination,
        best_log_probability,
        probability_sum,
        visited,
    )


def self_checks():
    print(
        "\nPokrećem matematičke "
        "i numeričke samoprovere.",
        flush=True,
    )

    # Independent reference for the corrected determinant wrapper.
    reference_matrix = np.diag(
        np.array(
            [1.0 + 1.0j, 2.0 - 1.0j],
            dtype=np.complex128,
        )
    )

    reference_sign, reference_log = checked_slogdet(
        reference_matrix
    )

    # (1+i)(2-i) = 3+i; |3+i| = sqrt(10).
    expected_sign = (
        (3.0 + 1.0j) / math.sqrt(10.0)
    )
    expected_log = 0.5 * math.log(10.0)

    require(
        abs(reference_sign - expected_sign) < 1e-12
        and abs(reference_log - expected_log) < 1e-12,
        "Referentna provera slogdet nije prošla.",
    )

    singular_matrix = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.complex128,
    )

    zero_sign, zero_log = checked_slogdet(
        singular_matrix
    )

    require(
        zero_sign == 0
        and np.isneginf(zero_log),
        "Provera nulte determinante nije prošla.",
    )

    number_count = 9
    draw_size = 3
    feature_count = 2

    b = initial_basis(
        number_count,
        draw_size,
        phase_order=2,
    )

    w = 0.025 * np.sin(
        np.arange(
            feature_count * number_count
        ).reshape(
            feature_count,
            number_count,
        )
        + 0.3
    )

    x = np.array(
        [
            [0.2, -0.4],
            [-0.3, 0.1],
            [0.7, 0.5],
            [-0.6, -0.2],
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [
            [0, 2, 5],
            [1, 4, 8],
            [0, 3, 7],
            [2, 6, 8],
        ],
        dtype=np.int64,
    )

    parameters = pack_parameters(
        b,
        w,
    )

    _, analytic = objective(
        parameters,
        x,
        targets,
        0.03,
        number_count,
        draw_size,
    )

    epsilon = 1e-6
    numerical = np.empty_like(parameters)

    for index in range(len(parameters)):
        plus = parameters.copy()
        minus = parameters.copy()

        plus[index] += epsilon
        minus[index] -= epsilon

        f_plus = objective(
            plus,
            x,
            targets,
            0.03,
            number_count,
            draw_size,
        )[0]

        f_minus = objective(
            minus,
            x,
            targets,
            0.03,
            number_count,
            draw_size,
        )[0]

        numerical[index] = (
            (f_plus - f_minus)
            / (2.0 * epsilon)
        )

    gradient_error = float(
        np.max(
            np.abs(analytic - numerical)
            / (
                1.0
                + np.abs(analytic)
                + np.abs(numerical)
            )
        )
    )

    require(
        gradient_error < 2e-5,
        "Provera izvoda nije prošla: "
        f"{gradient_error:.3e}.",
    )

    all_sets = np.asarray(
        list(
            combinations(
                range(number_count),
                draw_size,
            )
        ),
        dtype=np.int64,
    )

    repeated_x = np.repeat(
        x[:1],
        len(all_sets),
        axis=0,
    )

    log_probability = distribution_terms(
        b,
        w,
        repeated_x,
        all_sets,
    )[0]

    normalization = float(
        np.exp(
            logsumexp(log_probability)
        )
    )

    require(
        abs(normalization - 1.0) < 1e-10,
        "Mala potpuna provera normalizacije "
        "nije prošla.",
    )

    terms = distribution_terms(
        b,
        w,
        x,
        targets,
    )

    weights = terms[1]
    gram = terms[2]

    inverse_gram = np.linalg.inv(gram)

    b_inverse_gram = np.einsum(
        "ia,nab->nib",
        b,
        inverse_gram,
        optimize=True,
    )

    occupation = weights * np.einsum(
        "nib,ib->ni",
        b_inverse_gram,
        b.conj(),
        optimize=True,
    ).real

    require(
        np.max(
            np.abs(
                occupation.sum(axis=1)
                - draw_size
            )
        ) < 1e-9,
        "Zbir fermionskih zauzeća nije "
        "jednak broju izabranih stanja.",
    )

    require(
        np.min(occupation) >= -1e-9
        and np.max(occupation) <= 1.0 + 1e-9,
        "Fermionsko zauzeće je van opsega [0,1].",
    )

    duplicate = b[[1, 1, 4], :]

    _, duplicate_log_abs = checked_slogdet(
        duplicate
    )

    require(
        duplicate_log_abs < math.log(1e-12),
        "Provera Paulijevog isključivanja nije prošla.",
    )

    h, _ = log_amplitudes(
        x[:1],
        w,
    )

    q = orthonormal_basis(
        np.exp(h[0])[:, None] * b
    )

    (
        best,
        best_log_probability,
        _,
        _,
    ) = exhaustive_maximum(
        q,
        verbose=False,
    )

    direct_best = int(
        np.argmax(log_probability)
    )

    require(
        abs(
            best_log_probability
            - float(log_probability[direct_best])
        ) < 1e-9,
        "Provera izbora maksimuma nije prošla.",
    )

    matches = np.flatnonzero(
        np.all(
            all_sets == np.asarray(best),
            axis=1,
        )
    )

    require(
        len(matches) == 1,
        "Vraćena kombinacija nije pronađena "
        "u potpunoj proveri.",
    )

    require(
        abs(
            best_log_probability
            - float(
                log_probability[matches[0]]
            )
        ) < 1e-9,
        "Maksimum i vraćena kombinacija "
        "nisu saglasni.",
    )

    toy_draws = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=np.int64,
    )

    context = make_context(
        toy_draws,
        [1],
        lookback=3,
    )

    require(
        context[0, 39] == 1.0,
        "Prvi lag nije dostupan.",
    )
    require(
        context[0, 79] == 0.0
        and context[0, 119] == 0.0,
        "Nedostajući lag je pogrešno označen.",
    )
    require(
        float(context.sum()) == 8.0,
        "Kontekst prvog prelaza nije ispravan.",
    )

    print(
        "Samoprovere prošle: slogdet, "
        "nulta determinanta, izvod, normalizacija, "
        "zauzeća, isključivanje, maksimum "
        "i početni prelaz.\n"
        f"Relativna greška izvoda: "
        f"{gradient_error:.3e}",
        flush=True,
    )


def longest_consecutive_run(combination):
    longest = 1
    current = 1

    for left, right in zip(
        combination,
        combination[1:],
    ):
        if right == left + 1:
            current += 1
            longest = max(
                longest,
                current,
            )
        else:
            current = 1

    return longest


def process_file(
    path,
    draws,
    fingerprint,
):
    targets = draws[1:].copy()
    sample_count = len(targets)

    train_60 = int(
        0.60 * sample_count
    )
    train_70 = int(
        0.70 * sample_count
    )
    train_80 = int(
        0.80 * sample_count
    )

    require(
        50
        < train_60
        < train_70
        < train_80
        < sample_count,
        "Premalo podataka za dogovorenu "
        "hronološku podelu.",
    )

    contexts = {}

    for lookback, _ in CANDIDATES:
        if lookback not in contexts:
            contexts[lookback] = make_context(
                draws,
                np.arange(
                    1,
                    len(draws),
                    dtype=np.int64,
                ),
                lookback,
            )

    folds = (
        (train_60, train_70),
        (train_70, train_80),
    )

    scores = []

    for lookback, ridge in CANDIDATES:
        raw = contexts[lookback]
        total_nll = 0.0
        total_count = 0

        for fold_number, (
            train_end,
            validation_end,
        ) in enumerate(
            folds,
            start=1,
        ):
            model = fit_model(
                raw[:train_end],
                targets[:train_end],
                lookback,
                ridge,
                label=(
                    f"{path.name} / "
                    f"validacija {fold_number}"
                ),
            )

            count = (
                validation_end - train_end
            )

            nll = evaluate_nll(
                model,
                raw[train_end:validation_end],
                targets[train_end:validation_end],
            )

            total_nll += count * nll
            total_count += count

            print(
                "  Validacioni NLL po kolu: "
                f"{nll:.8f}",
                flush=True,
            )

        score = total_nll / total_count

        scores.append(
            (
                score,
                lookback,
                ridge,
            )
        )

        print(
            f"KANDIDAT: lag={lookback}, "
            f"lambda={ridge}, "
            "zbirni validacioni NLL="
            f"{score:.8f}",
            flush=True,
        )

    (
        validation_nll,
        selected_lookback,
        selected_ridge,
    ) = min(
        scores,
        key=lambda item: item[0],
    )

    raw = contexts[selected_lookback]

    print(
        f"\nIZABRANO: lag={selected_lookback}, "
        f"lambda={selected_ridge}, "
        f"validacioni NLL={validation_nll:.8f}",
        flush=True,
    )

    test_model = fit_model(
        raw[:train_80],
        targets[:train_80],
        selected_lookback,
        selected_ridge,
        label=(
            f"{path.name} / odvojeni test"
        ),
    )

    test_nll = evaluate_nll(
        test_model,
        raw[train_80:],
        targets[train_80:],
    )

    print(
        "ODVOJENI TEST: "
        f"{sample_count - train_80} kola, "
        f"NLL po kolu={test_nll:.8f}\n"
        "Test se ne koristi za promenu parametara.",
        flush=True,
    )

    final_model = fit_model(
        raw,
        targets,
        selected_lookback,
        selected_ridge,
        label=(
            f"{path.name} / završno učenje"
        ),
    )

    print(
        "ZAVRŠNO UČENJE: "
        f"svih {sample_count} prelaza "
        f"iz {len(draws)} kola.",
        flush=True,
    )

    raw_next = make_context(
        draws,
        [len(draws)],
        selected_lookback,
    )[0]

    q_next = final_model.basis(raw_next)

    next_projector = (
        q_next @ q_next.conj().T
    )

    context_sensitivity = 0.0

    # Diagnostic only. It never changes the selected combination.
    for raw_previous in raw[-8:]:
        q_previous = final_model.basis(
            raw_previous
        )

        previous_projector = (
            q_previous @ q_previous.conj().T
        )

        difference = float(
            np.linalg.norm(
                next_projector
                - previous_projector
            )
        )

        context_sensitivity = max(
            context_sensitivity,
            difference,
        )

    (
        best,
        log_probability,
        probability_sum,
        visited,
    ) = exhaustive_maximum(
        q_next,
        verbose=True,
        label=path.name,
    )

    require(
        len(best) == DRAW_SIZE,
        "NEXT nema sedam brojeva.",
    )
    require(
        len(set(best)) == DRAW_SIZE,
        "NEXT sadrži duplikate.",
    )
    require(
        all(
            0 <= value < NUMBER_COUNT
            for value in best
        ),
        "NEXT ima broj van opsega.",
    )

    run_length = longest_consecutive_run(best)

    historical_repetitions = int(
        np.count_nonzero(
            np.all(
                draws
                == np.asarray(best)[None, :],
                axis=1,
            )
        )
    )

    warnings = []

    if run_length >= 5:
        warnings.append(
            "NEXT sadrži niz od "
            f"{run_length} uzastopnih brojeva. "
            "Potreban je pregled modela; "
            "rezultat nije filtriran."
        )

    if historical_repetitions:
        warnings.append(
            "NEXT se pojavljuje "
            f"{historical_repetitions} puta "
            "u istoriji. Rezultat nije promenjen."
        )

    if context_sensitivity < 1e-8:
        warnings.append(
            "Nije utvrđena numerička promena "
            "izlaznog potprostora za proverene "
            "prethodne kontekste. Potrebna je "
            "provera uticaja istorijskog ulaza."
        )

    return {
        "name": path.name,
        "fingerprint": fingerprint,
        "combination": tuple(
            value + 1 for value in best
        ),
        "validation_nll": validation_nll,
        "test_nll": test_nll,
        "model_probability": math.exp(
            log_probability
        ),
        "probability_sum": probability_sum,
        "visited": visited,
        "context_sensitivity": context_sensitivity,
        "run_length": run_length,
        "span": best[-1] - best[0],
        "warnings": warnings,
    }


def main():
    np.seterr(
        over="raise",
        invalid="raise",
        divide="raise",
    )

    print(
        "V1 FERMION — Loto 7/39\n"
        "Lokalni proračun kompleksnih "
        "fermionskih amplituda.\n"
        "Bez slučajnog generisanja "
        "i bez klasifikatora.\n"
        "Lokalna obrada slogdet je uključena.",
        flush=True,
    )

    resolved_paths = [
        path.resolve()
        for path, _ in FILES
    ]

    require(
        len(set(resolved_paths)) == len(FILES),
        "Putanje CSV-ova nisu različite.",
    )

    datasets = []

    for path, expected_rows in FILES:
        draws, fingerprint = load_draws(
            path,
            expected_rows,
        )

        datasets.append(
            (
                path,
                draws,
                fingerprint,
            )
        )

    require(
        len(
            {
                fingerprint
                for _, _, fingerprint in datasets
            }
        ) == 3,
        "Dva ulazna fajla imaju identičan sadržaj.",
    )

    self_checks()

    results = []

    for path, draws, fingerprint in datasets:
        results.append(
            process_file(
                path,
                draws,
                fingerprint,
            )
        )

    for left in range(len(results)):
        for right in range(
            left + 1,
            len(results),
        ):
            if (
                results[left]["combination"]
                == results[right]["combination"]
            ):
                message = (
                    "Ista NEXT kombinacija za "
                    "različite CSV-ove: "
                    f"{results[left]['name']} i "
                    f"{results[right]['name']}. "
                    "Proveriti uticaj istorije "
                    "i moguću degeneraciju. "
                    "Kombinacija nije promenjena."
                )

                results[left]["warnings"].append(
                    message
                )
                results[right]["warnings"].append(
                    message
                )

    print(
        "\n"
        "============================================================\n"
        "ZAVRŠNI REZULTATI V1 FERMION\n"
        "============================================================\n"
        "Modelna verovatnoća pripada naučenom modelu.\n"
        "Ona nije potvrđena verovatnoća stvarnog "
        "narednog izvlačenja.\n"
        "NLL je negativna log-verovatnoća "
        "u prirodnim logaritmima.\n"
        "Niži NLL označava bolji rezultat "
        "na tom skupu.",
        flush=True,
    )

    for result in results:
        numbers = " ".join(
            f"{value:02d}"
            for value in result["combination"]
        )

        print(
            f"\nCSV: {result['name']}\n"
            f"NEXT: {numbers}\n"
            "Validacioni NLL: "
            f"{result['validation_nll']:.8f}\n"
            "Odvojeni test NLL: "
            f"{result['test_nll']:.8f}\n"
            "Modelna verovatnoća NEXT: "
            f"{result['model_probability']:.12g}\n"
            "Pregledano kombinacija: "
            f"{result['visited']:,}\n"
            "Zbir svih verovatnoća: "
            f"{result['probability_sum']:.12f}\n"
            "Promena potprostora prema "
            "prethodnim kontekstima: "
            f"{result['context_sensitivity']:.6e}\n"
            "Najduži uzastopni niz: "
            f"{result['run_length']}\n"
            "Raspon kombinacije: "
            f"{result['span']}",
            flush=True,
        )

        for warning in result["warnings"]:
            print(
                f"PROVERA MODELA: {warning}",
                flush=True,
            )

    print(
        "\nZavršna pretraga je potpuna "
        "za svaki prihvaćeni model.\n"
        "Učenje je lokalna numerička optimizacija. "
        "Globalni optimum parametara nije dokazan.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()

    except (
        OSError,
        UnicodeError,
        csv.Error,
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        print(
            f"\nZAUSTAVLJENO: {error}\n"
            "Završni skup od tri NEXT "
            "rezultata nije potvrđen.\n"
            "Tačno mesto prekida:",
            file=sys.stderr,
            flush=True,
        )

        traceback.print_exc()
        sys.exit(1)



"""
/Pauli_fermion_boson/Pauli_fermion_V1.py:1892: 
V1 FERMION — Loto 7/39
Lokalni proračun kompleksnih fermionskih amplituda.
Bez slučajnog generisanja i bez klasifikatora.
Lokalna obrada slogdet je uključena.

ULAZ: loto7_4710_k86.csv
Redova: 4710 | praznih redova: 0
SHA256: e51a5ab684401459535996f50fd0b0540b0af6d2b86bca49825cf381d759c85d
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_2977.csv
Redova: 2977 | praznih redova: 0
SHA256: 69ae55edbff87383f9bb4a25df4ba6b44acbf23404db6e7fa3642de0442479f4
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_plus_1733.csv
Redova: 1733 | praznih redova: 0
SHA256: 9d708eb88434b55a38bbc69474a869955e0295f1bed401cf10dc177d7d8fef39
Redosled kola ostaje redosled iz CSV-a.

Pokrećem matematičke i numeričke samoprovere.
Samoprovere prošle: slogdet, nulta determinanta, izvod, normalizacija, zauzeća, isključivanje, maksimum i početni prelaz.
Relativna greška izvoda: 5.787e-10
loto7_4710_k86.csv / validacija 1: lag=1, lambda=0.03, početne faze=17, trening=2825
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.41119845, max|grad|=9.933e-07
loto7_4710_k86.csv / validacija 1: lag=1, lambda=0.03, početne faze=11, trening=2825
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.42477994, max|grad|=8.663e-07
  Validacioni NLL po kolu: 17.20745542
loto7_4710_k86.csv / validacija 2: lag=1, lambda=0.03, početne faze=17, trening=3296
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.49604527, max|grad|=7.749e-07
loto7_4710_k86.csv / validacija 2: lag=1, lambda=0.03, početne faze=11, trening=3296
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Optimizacija: 400 iteracija
  Prihvaćen lokalni minimum: gubitak=16.50060615, max|grad|=4.845e-07
  Validacioni NLL po kolu: 17.33176671
KANDIDAT: lag=1, lambda=0.03, zbirni validacioni NLL=17.26961107
loto7_4710_k86.csv / validacija 1: lag=1, lambda=0.3, početne faze=17, trening=2825
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.46231838, max|grad|=6.045e-07
loto7_4710_k86.csv / validacija 1: lag=1, lambda=0.3, početne faze=11, trening=2825
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Optimizacija: 400 iteracija
  Prihvaćen lokalni minimum: gubitak=16.44600776, max|grad|=4.745e-07
  Validacioni NLL po kolu: 17.33696722
loto7_4710_k86.csv / validacija 2: lag=1, lambda=0.3, početne faze=17, trening=3296
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.49560469, max|grad|=9.365e-07
loto7_4710_k86.csv / validacija 2: lag=1, lambda=0.3, početne faze=11, trening=3296
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.49775842, max|grad|=7.481e-07
  Validacioni NLL po kolu: 17.32524438
KANDIDAT: lag=1, lambda=0.3, zbirni validacioni NLL=17.33110580
loto7_4710_k86.csv / validacija 1: lag=3, lambda=0.03, početne faze=17, trening=2825
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.41815975, max|grad|=6.968e-07
loto7_4710_k86.csv / validacija 1: lag=3, lambda=0.03, početne faze=11, trening=2825
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Optimizacija: 400 iteracija
  Prihvaćen lokalni minimum: gubitak=16.41548499, max|grad|=6.629e-07
  Validacioni NLL po kolu: 17.34018822
loto7_4710_k86.csv / validacija 2: lag=3, lambda=0.03, početne faze=17, trening=3296
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Optimizacija: 400 iteracija
  Prihvaćen lokalni minimum: gubitak=16.48397253, max|grad|=9.141e-07
loto7_4710_k86.csv / validacija 2: lag=3, lambda=0.03, početne faze=11, trening=3296
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Optimizacija: 400 iteracija
  Prihvaćen lokalni minimum: gubitak=16.49911250, max|grad|=9.759e-07
  Validacioni NLL po kolu: 17.28342120
KANDIDAT: lag=3, lambda=0.03, zbirni validacioni NLL=17.31180471
loto7_4710_k86.csv / validacija 1: lag=3, lambda=0.3, početne faze=17, trening=2825
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.44398899, max|grad|=6.251e-07
loto7_4710_k86.csv / validacija 1: lag=3, lambda=0.3, početne faze=11, trening=2825
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.46681795, max|grad|=6.982e-07
  Validacioni NLL po kolu: 17.13571352
loto7_4710_k86.csv / validacija 2: lag=3, lambda=0.3, početne faze=17, trening=3296
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Optimizacija: 400 iteracija
  Prihvaćen lokalni minimum: gubitak=16.47755723, max|grad|=8.946e-07
loto7_4710_k86.csv / validacija 2: lag=3, lambda=0.3, početne faze=11, trening=3296
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.54397072, max|grad|=5.952e-07
  Validacioni NLL po kolu: 17.23353577
KANDIDAT: lag=3, lambda=0.3, zbirni validacioni NLL=17.18462465

IZABRANO: lag=3, lambda=0.3, validacioni NLL=17.18462465
loto7_4710_k86.csv / odvojeni test: lag=3, lambda=0.3, početne faze=17, trening=3767
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.55660266, max|grad|=6.767e-07
loto7_4710_k86.csv / odvojeni test: lag=3, lambda=0.3, početne faze=11, trening=3767
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.56460607, max|grad|=8.394e-07
ODVOJENI TEST: 942 kola, NLL po kolu=17.28842909
Test se ne koristi za promenu parametara.
loto7_4710_k86.csv / završno učenje: lag=3, lambda=0.3, početne faze=17, trening=4709
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Optimizacija: 400 iteracija
  Prihvaćen lokalni minimum: gubitak=16.61127652, max|grad|=6.217e-07
loto7_4710_k86.csv / završno učenje: lag=3, lambda=0.3, početne faze=11, trening=4709
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.62837172, max|grad|=7.812e-07
ZAVRŠNO UČENJE: svih 4709 prelaza iz 4710 kola.
loto7_4710_k86.csv: pregledano 2,002,944/15,380,937
loto7_4710_k86.csv: pregledano 4,001,792/15,380,937
loto7_4710_k86.csv: pregledano 6,000,640/15,380,937
loto7_4710_k86.csv: pregledano 8,003,584/15,380,937
loto7_4710_k86.csv: pregledano 10,002,432/15,380,937
loto7_4710_k86.csv: pregledano 12,001,280/15,380,937
loto7_4710_k86.csv: pregledano 14,000,128/15,380,937



loto7_4710_k86_loto_2977.csv / validacija 1: lag=1, lambda=0.03, početne faze=17, trening=1785
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.24241783, max|grad|=9.765e-07
loto7_4710_k86_loto_2977.csv / validacija 1: lag=1, lambda=0.03, početne faze=11, trening=1785
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.21075331, max|grad|=6.558e-07
  Validacioni NLL po kolu: 17.33291075
loto7_4710_k86_loto_2977.csv / validacija 2: lag=1, lambda=0.03, početne faze=17, trening=2083
  Optimizacija: 100 iteracija
  Prihvaćen lokalni minimum: gubitak=16.31024548, max|grad|=9.579e-07
loto7_4710_k86_loto_2977.csv / validacija 2: lag=1, lambda=0.03, početne faze=11, trening=2083
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.29098008, max|grad|=5.108e-07
  Validacioni NLL po kolu: 17.23789522
KANDIDAT: lag=1, lambda=0.03, zbirni validacioni NLL=17.28548283
loto7_4710_k86_loto_2977.csv / validacija 1: lag=1, lambda=0.3, početne faze=17, trening=1785
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.21654676, max|grad|=9.811e-07
loto7_4710_k86_loto_2977.csv / validacija 1: lag=1, lambda=0.3, početne faze=11, trening=1785
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.21812566, max|grad|=5.177e-07
  Validacioni NLL po kolu: 17.20587240
loto7_4710_k86_loto_2977.csv / validacija 2: lag=1, lambda=0.3, početne faze=17, trening=2083
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.28672787, max|grad|=7.873e-07
loto7_4710_k86_loto_2977.csv / validacija 2: lag=1, lambda=0.3, početne faze=11, trening=2083
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.30975093, max|grad|=4.212e-07
  Validacioni NLL po kolu: 17.31503953
KANDIDAT: lag=1, lambda=0.3, zbirni validacioni NLL=17.26036423
loto7_4710_k86_loto_2977.csv / validacija 1: lag=3, lambda=0.03, početne faze=17, trening=1785
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.20116428, max|grad|=6.303e-07
loto7_4710_k86_loto_2977.csv / validacija 1: lag=3, lambda=0.03, početne faze=11, trening=1785
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.20754243, max|grad|=6.124e-07
  Validacioni NLL po kolu: 17.40433040
loto7_4710_k86_loto_2977.csv / validacija 2: lag=3, lambda=0.03, početne faze=17, trening=2083
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.27476432, max|grad|=5.573e-07
loto7_4710_k86_loto_2977.csv / validacija 2: lag=3, lambda=0.03, početne faze=11, trening=2083
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.26622518, max|grad|=8.536e-07
  Validacioni NLL po kolu: 17.43124214
KANDIDAT: lag=3, lambda=0.03, zbirni validacioni NLL=17.41776366
loto7_4710_k86_loto_2977.csv / validacija 1: lag=3, lambda=0.3, početne faze=17, trening=1785
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.21827045, max|grad|=6.179e-07
loto7_4710_k86_loto_2977.csv / validacija 1: lag=3, lambda=0.3, početne faze=11, trening=1785
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.25088292, max|grad|=8.296e-07
  Validacioni NLL po kolu: 17.32288724
loto7_4710_k86_loto_2977.csv / validacija 2: lag=3, lambda=0.3, početne faze=17, trening=2083
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.29331847, max|grad|=7.128e-07
loto7_4710_k86_loto_2977.csv / validacija 2: lag=3, lambda=0.3, početne faze=11, trening=2083
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.30986921, max|grad|=7.257e-07
  Validacioni NLL po kolu: 17.31596052
KANDIDAT: lag=3, lambda=0.3, zbirni validacioni NLL=17.31942970

IZABRANO: lag=1, lambda=0.3, validacioni NLL=17.26036423
loto7_4710_k86_loto_2977.csv / odvojeni test: lag=1, lambda=0.3, početne faze=17, trening=2380
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Optimizacija: 400 iteracija
  Prihvaćen lokalni minimum: gubitak=16.36015304, max|grad|=5.689e-07
loto7_4710_k86_loto_2977.csv / odvojeni test: lag=1, lambda=0.3, početne faze=11, trening=2380
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.36587441, max|grad|=7.541e-07
ODVOJENI TEST: 596 kola, NLL po kolu=17.30834659
Test se ne koristi za promenu parametara.
loto7_4710_k86_loto_2977.csv / završno učenje: lag=1, lambda=0.3, početne faze=17, trening=2976
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.48980850, max|grad|=8.514e-07
loto7_4710_k86_loto_2977.csv / završno učenje: lag=1, lambda=0.3, početne faze=11, trening=2976
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Optimizacija: 400 iteracija
  Prihvaćen lokalni minimum: gubitak=16.45131590, max|grad|=9.205e-07
ZAVRŠNO UČENJE: svih 2976 prelaza iz 2977 kola.
loto7_4710_k86_loto_2977.csv: pregledano 2,002,944/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 4,001,792/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 6,000,640/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 8,003,584/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 10,002,432/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 12,001,280/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 14,000,128/15,380,937



loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=1, lambda=0.03, početne faze=17, trening=1039
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=15.88510021, max|grad|=9.202e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=1, lambda=0.03, početne faze=11, trening=1039
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=15.91867638, max|grad|=9.233e-07
  Validacioni NLL po kolu: 17.47366375
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=1, lambda=0.03, početne faze=17, trening=1212
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=15.99167529, max|grad|=8.057e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=1, lambda=0.03, početne faze=11, trening=1212
  Optimizacija: 100 iteracija
  Prihvaćen lokalni minimum: gubitak=15.99830536, max|grad|=8.571e-07
  Validacioni NLL po kolu: 17.52062488
KANDIDAT: lag=1, lambda=0.03, zbirni validacioni NLL=17.49714432
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=1, lambda=0.3, početne faze=17, trening=1039
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Optimizacija: 400 iteracija
  Prihvaćen lokalni minimum: gubitak=15.93382725, max|grad|=6.320e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=1, lambda=0.3, početne faze=11, trening=1039
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=15.94523390, max|grad|=9.840e-07
  Validacioni NLL po kolu: 17.37541067
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=1, lambda=0.3, početne faze=17, trening=1212
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Optimizacija: 400 iteracija
  Optimizacija: 500 iteracija
  Prihvaćen lokalni minimum: gubitak=16.01608595, max|grad|=7.496e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=1, lambda=0.3, početne faze=11, trening=1212
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.04491515, max|grad|=5.149e-07
  Validacioni NLL po kolu: 17.36313301
KANDIDAT: lag=1, lambda=0.3, zbirni validacioni NLL=17.36927184
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=3, lambda=0.03, početne faze=17, trening=1039
  Optimizacija: 100 iteracija
  Prihvaćen lokalni minimum: gubitak=15.92076356, max|grad|=5.623e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=3, lambda=0.03, početne faze=11, trening=1039
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=15.88503044, max|grad|=5.582e-07
  Validacioni NLL po kolu: 17.44368277
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=3, lambda=0.03, početne faze=17, trening=1212
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=15.98613164, max|grad|=9.505e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=3, lambda=0.03, početne faze=11, trening=1212
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=15.97219472, max|grad|=6.451e-07
  Validacioni NLL po kolu: 17.18650525
KANDIDAT: lag=3, lambda=0.03, zbirni validacioni NLL=17.31509401
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=3, lambda=0.3, početne faze=17, trening=1039
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=15.94944272, max|grad|=8.070e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=3, lambda=0.3, početne faze=11, trening=1039
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=15.93889237, max|grad|=7.169e-07
  Validacioni NLL po kolu: 17.30720820
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=3, lambda=0.3, početne faze=17, trening=1212
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.03398938, max|grad|=5.471e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=3, lambda=0.3, početne faze=11, trening=1212
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.05705303, max|grad|=9.002e-07
  Validacioni NLL po kolu: 17.45309076
KANDIDAT: lag=3, lambda=0.3, zbirni validacioni NLL=17.38014948

IZABRANO: lag=3, lambda=0.03, validacioni NLL=17.31509401
loto7_4710_k86_loto_plus_1733.csv / odvojeni test: lag=3, lambda=0.03, početne faze=17, trening=1385
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.11619008, max|grad|=9.622e-07
loto7_4710_k86_loto_plus_1733.csv / odvojeni test: lag=3, lambda=0.03, početne faze=11, trening=1385
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.05294872, max|grad|=9.861e-07
ODVOJENI TEST: 347 kola, NLL po kolu=17.30045856
Test se ne koristi za promenu parametara.
loto7_4710_k86_loto_plus_1733.csv / završno učenje: lag=3, lambda=0.03, početne faze=17, trening=1732
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.19559650, max|grad|=9.231e-07
loto7_4710_k86_loto_plus_1733.csv / završno učenje: lag=3, lambda=0.03, početne faze=11, trening=1732
  Optimizacija: 100 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.18959723, max|grad|=8.857e-07
ZAVRŠNO UČENJE: svih 1732 prelaza iz 1733 kola.
loto7_4710_k86_loto_plus_1733.csv: pregledano 2,002,944/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 4,001,792/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 6,000,640/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 8,003,584/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 10,002,432/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 12,001,280/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 14,000,128/15,380,937



============================================================
ZAVRŠNI REZULTATI V1 FERMION
============================================================
Modelna verovatnoća pripada naučenom modelu.
Ona nije potvrđena verovatnoća stvarnog narednog izvlačenja.
NLL je negativna log-verovatnoća u prirodnim logaritmima.
Niži NLL označava bolji rezultat na tom skupu.

CSV: loto7_4710_k86.csv
NEXT: 06 x 13 y 28 z 37
Validacioni NLL: 17.18462465
Odvojeni test NLL: 17.28842909
Modelna verovatnoća NEXT: 2.57576645255e-06
Pregledano kombinacija: 15,380,937
Zbir svih verovatnoća: 1.000000000000
Promena potprostora prema prethodnim kontekstima: 1.476224e-01
Najduži uzastopni niz: 1
Raspon kombinacije: 31

CSV: loto7_4710_k86_loto_2977.csv
NEXT: 08 x 12 y 22 z 27
Validacioni NLL: 17.26036423
Odvojeni test NLL: 17.30834659
Modelna verovatnoća NEXT: 1.98394798592e-06
Pregledano kombinacija: 15,380,937
Zbir svih verovatnoća: 1.000000000000
Promena potprostora prema prethodnim kontekstima: 2.182804e-01
Najduži uzastopni niz: 2
Raspon kombinacije: 19

CSV: loto7_4710_k86_loto_plus_1733.csv
NEXT: 02 x 08 y 15 z 33
Validacioni NLL: 17.31509401
Odvojeni test NLL: 17.30045856
Modelna verovatnoća NEXT: 2.69673113843e-06
Pregledano kombinacija: 15,380,937
Zbir svih verovatnoća: 1.000000000000
Promena potprostora prema prethodnim kontekstima: 4.509715e-01
Najduži uzastopni niz: 2
Raspon kombinacije: 31

Završna pretraga je potpuna za svaki prihvaćeni model.
Učenje je lokalna numerička optimizacija. 
Globalni optimum parametara nije dokazan.
"""



"""
fermion: antisimetrične amplitude, Slaterova determinanta i Paulijevo isključivanje 

Paulijev princip može se koristiti kao osnova modela raspodele za Loto 7/39. 
Sam princip ne određuje koji će brojevi izaći. 
Njegova korisna osobina je matematičko isključivanje ponavljanja unutar jedne kombinacije.

Prenosimo iz fizike: 
Paulijev princip kaže da dva identična fermiona ne mogu zauzeti isto potpuno određeno kvantno stanje. 
Antisimetrična amplituda opisuje to isključivanje. 
Za moju adaptaciju broj od 1 do 39 predstavlja jedno dozvoljeno stanje. 
To je matematička analogija; loto kuglice ne tretiram kao fermione. 

Kombinaciju predstavljam sa sedam zauzetih stanja među 39:
\[
n_i\in\{0,1\},\qquad \sum_{i=1}^{39}n_i=7.
\]
Time dobijam tačno sedam različitih brojeva, bez popravljanja duplikata.

Model: naučena antisimetrična amplituda

Za svako naredno kolo model bi iz prethodne istorije računao matricu amplituda \(A_t\), dimenzije \(39\times7\).

Za kombinaciju \(S\) od sedam brojeva izdvajam njenih sedam redova. 
Amplituda kombinacije je Slaterova determinanta:
\[
\psi_t(S)=\det(A_t[S,:]).
\]
Raspodelu kombinacija definišem kao:
\[
P_t(S)=
\frac{|\det(A_t[S,:])|^2}
{\det(A_t^\dagger A_t)},
\qquad |S|=7.
\]
Ova normalizacija važi kada matrica ima rang sedam. 
Oznaka \(\dagger\) znači konjugovano transponovanje. 
Ako se broj ponovi, determinanta ima dva jednaka reda i amplituda je nula.

Ovakva konstrukcija povezana je sa determinantalnim procesima fiksne veličine, odnosno 7-DPP modelima. 
Oni opisuju raspodelu nad skupovima sa tačno sedam elemenata. 

Model mora da nauč: 
Paulijevo isključivanje daje strukturu. 
Prediktivni deo mora da nauči amplitude iz istorijskih prelaza između kola. 
Predlažem uslovnu regresiju raspodele: prethodna kola su ulaz, a raspodela naredne kombinacije je izlaz.

Jedna Slaterova determinanta ograničava vrste zavisnosti koje model može naučiti. 
Determinantalna konstrukcija favorizuje različitost u naučenom prostoru karakteristika. 
Zato ne smem unapred postaviti da su numerički bliski brojevi međusobno nepoželjni. 
To bi nametnulo pravilo o razmacima koje nisam odobrio.

Istraživanje kao amplitudni model raspodele. 
Sama zabrana duplikata nije dovoljan doprinos. 
Vrednost mora pokazati naučena raspodela na odvojenom hronološkom testu.

Navedena formula može se računati klasično; njeno računanje samo po sebi nije kvantno kolo.

V1 — fermionski model za sva tri ažurirana CSV-a. 
V2 ostaje za bozonski model za kasnije. 
Učenje amplituda, hronološka validacija i izbor NEXT kombinacije.

Fajlovi imaju 4710, 2977 i 1733 neprazna reda. Početak i kraj svakog fajla imaju po sedam brojeva, bez zaglavlja.

V1 zasnovan na kompleksnim amplitudama i Slaterovoj determinanti. 
Parametri će se učiti iz istorijskih prelaza. 
Završni izbor će pregledati svih 15.380.937 kombinacija i proveriti zbir njihovih verovatnoća. 
To je lokalni klasični proračun fermionskih amplituda, bez Qiskit kola. 

Konstrukcija čuva sedam različitih brojeva kroz determinantu. 
Završni model koristi sve raspoložive prelaze, uključujući početne prelaze sa kraćom prethodnom istorijom. 
U kod uključujem proveru izvoda funkcije gubitka, normalizacije i potpunosti završne pretrage. 

V1 koristi fermionske amplitude i Slaterovu determinantu. 
Ovo je lokalni klasični proračun amplituda, bez Qiskit kola.
"""
