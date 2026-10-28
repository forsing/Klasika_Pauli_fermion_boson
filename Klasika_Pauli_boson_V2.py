#!/usr/bin/env python3



"""
V2 BOSON — Loto 7/39
Python 3.11
Dependencies: numpy, scipy

Classical computation of symmetric bosonic amplitudes.
No quantum circuit or quantum hardware is used.

Two learned complex orbitals:
    u: occupation 4
    v: occupation 3

For a seven-number set S:

    c(S) = coefficient of z^3 in product(u_i + z*v_i, i in S)

The permanent of the corresponding matrix with four u columns
and three v columns equals:

    permanent = 4! * 3! * c(S)

The common factorial cancels in the normalized distribution.

Conditional orbitals:

    A_i(x) = exp(h_i(x)) * B_i
    h(x) = CAP * tanh((x W) / CAP)

Loto distribution:

    P(S | x) = |c_A(S)|^2 / Z(x)

    Z(x) = sum |c_A(T)|^2 over all seven-element subsets T

Z is computed by an exact coefficient recurrence.
The recurrence selects each number at most once.

Bosonic symmetry comes from the permanent.
Distinct output numbers come from conditioning on legal Loto sets.

No random initialization, sampling, shuffling or random splits.
No classifier.
No rounding, duplicate repair or cosmetic combination filters.

Chronological parameter selection:
    Train 60%; validate next 10%.
    Train 70%; validate next 10%.
    Select parameters by validation NLL.
    Train 80%; evaluate final 20% once.
    Fit the selected model on ALL available historical transitions.

NEXT:
    Exhaustive search over all 15,380,937 legal combinations.
    Independently sum their probabilities during that search.
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
from functools import lru_cache
from itertools import combinations, islice, permutations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp



FILES = (
    (
        Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86.csv"),
        4710,
    ),
    (
        Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86_loto_2977.csv"),
        2977,
    ),
    (
        Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86_loto_plus_1733.csv"),
        1733,
    ),
)



NUMBER_COUNT = 39
DRAW_SIZE = 7

U_COUNT = 4
V_COUNT = 3

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


def say(*lines):
    for line in lines:
        print(line, flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


@dataclass(frozen=True)
class BosonSpace:
    u_count: int
    v_count: int
    states: tuple
    edges: tuple
    start: int
    end: int

    @property
    def draw_size(self):
        return self.u_count + self.v_count

    @property
    def size(self):
        return len(self.states)


@lru_cache(maxsize=4)
def make_space(u_count, v_count):
    require(
        u_count >= 1 and v_count >= 1,
        "Obe orbitale moraju imati pozitivno zauzeće.",
    )

    draw_size = u_count + v_count
    states = []

    # State: counts of u/v in the conjugate and ordinary factors.
    # Both sides always have the same total occupation.
    for level in range(draw_size + 1):
        for a in range(u_count + 1):
            for b in range(v_count + 1):
                if a + b != level:
                    continue

                for c in range(u_count + 1):
                    for d in range(v_count + 1):
                        if c + d == level:
                            states.append((a, b, c, d))

    lookup = {
        state: index
        for index, state in enumerate(states)
    }

    edges = []

    for conjugate_orbital in range(2):
        for ordinary_orbital in range(2):
            sources = []
            destinations = []

            for index, state in enumerate(states):
                destination = list(state)
                destination[conjugate_orbital] += 1
                destination[2 + ordinary_orbital] += 1

                destination_index = lookup.get(
                    tuple(destination)
                )

                if destination_index is not None:
                    sources.append(index)
                    destinations.append(destination_index)

            edges.append(
                (
                    conjugate_orbital,
                    ordinary_orbital,
                    np.asarray(sources, dtype=np.int64),
                    np.asarray(destinations, dtype=np.int64),
                )
            )

    return BosonSpace(
        u_count=u_count,
        v_count=v_count,
        states=tuple(states),
        edges=tuple(edges),
        start=lookup[(0, 0, 0, 0)],
        end=lookup[
            (u_count, v_count, u_count, v_count)
        ],
    )


def normalize_columns(b):
    norms = np.sqrt(
        np.sum(np.abs(b) ** 2, axis=0)
    )

    require(
        np.all(np.isfinite(norms))
        and np.all(norms > 1e-12),
        "Orbitala ima nultu ili neispravnu normu.",
    )

    # Independent nonzero column scaling cancels from P(S|x).
    # Do not rotate or mix the two orbitals.
    return b / norms[None, :]


def initial_orbitals(number_count, phase_order):
    require(
        math.gcd(phase_order, number_count) == 1,
        "Neispravan deterministički redosled faza.",
    )

    labels = (
        phase_order
        * np.arange(number_count, dtype=np.int64)
        + 11
    ) % number_count

    angle = 2.0 * np.pi * labels / number_count

    u = (
        (1.0 + 0.15 * np.cos(angle))
        * np.exp(0.1j * np.sin(angle))
    )

    v = (
        (1.0 + 0.15 * np.sin(angle))
        * np.exp(0.1j * np.cos(angle))
    )

    return normalize_columns(
        np.column_stack((u, v))
    )


def boson_amplitude(
    selected,
    v_count,
    with_gradient=False,
):
    """
    Compute the coefficient of z^v_count in:
        product(u_i + z*v_i)

    This equals the permanent divided by:
        u_count! * v_count!

    Rows may be permuted without changing this amplitude.
    """
    selected = np.asarray(
        selected,
        dtype=np.complex128,
    )

    require(
        selected.ndim == 3
        and selected.shape[2] == 2,
        "Neispravan oblik izabranih orbitala.",
    )

    sample_count, draw_size, _ = selected.shape

    require(
        0 <= v_count <= draw_size,
        "Neispravno bozonsko zauzeće.",
    )
    require(
        np.all(np.isfinite(selected)),
        "Amplitudni ulaz nije konačan.",
    )

    polynomial = np.zeros(
        (sample_count, v_count + 1),
        dtype=np.complex128,
    )
    polynomial[:, 0] = 1.0

    cache = [polynomial] if with_gradient else None

    for position in range(draw_size):
        previous = polynomial

        polynomial = (
            previous
            * selected[:, position, 0][:, None]
        )

        if v_count:
            polynomial[:, 1:] += (
                previous[:, :-1]
                * selected[:, position, 1][:, None]
            )

        if with_gradient:
            cache.append(polynomial)

    amplitude = polynomial[:, v_count]

    require(
        np.all(np.isfinite(amplitude)),
        "Bozonska amplituda nije konačna.",
    )

    if not with_gradient:
        return amplitude, None

    reverse = np.zeros_like(polynomial)
    reverse[:, v_count] = 1.0

    derivative = np.zeros_like(selected)

    for position in range(draw_size - 1, -1, -1):
        previous = cache[position]

        derivative[:, position, 0] = np.sum(
            reverse * previous,
            axis=1,
        )

        if v_count:
            derivative[:, position, 1] = np.sum(
                reverse[:, 1:] * previous[:, :-1],
                axis=1,
            )

        previous_reverse = (
            reverse
            * selected[:, position, 0][:, None]
        )

        if v_count:
            previous_reverse[:, :-1] += (
                reverse[:, 1:]
                * selected[:, position, 1][:, None]
            )

        reverse = previous_reverse

    return amplitude, derivative


def boson_partition(
    b,
    weights,
    space,
    with_gradient=False,
):
    """
    Exact legal-set normalizer.

    Coefficient of:
        x^u y^v X^u Y^v

    in:
        product_i(
            1
            + weight_i
              * (conj(u_i)*x + conj(v_i)*y)
              * (u_i*X + v_i*Y)
        )

    The constant term means: do not select number i.
    The quadratic term means: select number i exactly once.

    With gradients:
        return derivative of sum(log Z) with respect to
        real and imaginary parts of B, packed as a complex array;
        return conditional occupation probabilities.
    """
    b = np.asarray(b, dtype=np.complex128)
    weights = np.asarray(weights, dtype=np.float64)

    require(
        b.ndim == 2
        and b.shape[1] == 2,
        "Neispravan oblik orbitala.",
    )
    require(
        weights.ndim == 2
        and weights.shape[1] == b.shape[0]
        and weights.shape[0] > 0,
        "Neispravan oblik težina normalizacije.",
    )
    require(
        np.all(np.isfinite(b))
        and np.all(np.isfinite(weights))
        and np.all(weights > 0),
        "Neispravne vrednosti u normalizaciji.",
    )

    sample_count, number_count = weights.shape

    forward = np.zeros(
        (sample_count, space.size),
        dtype=np.complex128,
    )
    forward[:, space.start] = 1.0

    cache = [forward] if with_gradient else None

    for number in range(number_count):
        following = forward.copy()

        for a, c, sources, destinations in space.edges:
            coefficient = (
                weights[:, number]
                * np.conj(b[number, a])
                * b[number, c]
            )

            following[:, destinations] += (
                forward[:, sources]
                * coefficient[:, None]
            )

        forward = following

        if with_gradient:
            cache.append(forward)

    complex_z = forward[:, space.end]

    require(
        np.all(np.isfinite(complex_z)),
        "Normalizaciona konstanta nije konačna.",
    )

    z = complex_z.real

    require(
        np.all(z > 0),
        "Normalizaciona konstanta nije pozitivna.",
    )
    require(
        np.all(
            np.abs(complex_z.imag)
            <= 1e-8 * z
        ),
        "Normalizacija ima preveliki imaginarni ostatak.",
    )

    log_z = np.log(z)

    if not with_gradient:
        return log_z, None, None

    reverse = np.zeros_like(forward)
    reverse[:, space.end] = 1.0 / z

    gradient_b = np.zeros_like(b)

    occupation = np.zeros(
        (sample_count, number_count),
        dtype=np.float64,
    )

    for number in range(number_count - 1, -1, -1):
        previous = cache[number]
        previous_reverse = reverse.copy()

        for a, c, sources, destinations in space.edges:
            edge_value = np.sum(
                previous[:, sources]
                * reverse[:, destinations],
                axis=1,
            )

            coefficient = (
                weights[:, number]
                * np.conj(b[number, a])
                * b[number, c]
            )

            occupation[:, number] += (
                coefficient * edge_value
            ).real

            gradient_b[number, a] += (
                2.0
                * np.sum(
                    weights[:, number]
                    * b[number, c]
                    * edge_value
                )
            )

            previous_reverse[:, sources] += (
                reverse[:, destinations]
                * coefficient[:, None]
            )

        reverse = previous_reverse

    require(
        np.all(np.isfinite(gradient_b))
        and np.all(np.isfinite(occupation)),
        "Izvod normalizacije nije konačan.",
    )
    require(
        np.max(
            np.abs(
                occupation.sum(axis=1)
                - space.draw_size
            )
        ) < 1e-5,
        "Zbir uslovnih zauzeća nije jednak veličini kombinacije.",
    )
    require(
        np.min(occupation) >= -1e-5
        and np.max(occupation) <= 1.0 + 1e-5,
        "Uslovno zauzeće legalnog skupa je van opsega [0,1].",
    )

    return log_z, gradient_b, occupation


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
                f"{path.name}, red {line_number}: potrebno je 7 kolona.",
            )

            try:
                numbers = [int(item.strip()) for item in row]
            except ValueError as error:
                raise RuntimeError(
                    f"{path.name}, red {line_number}: neispravan ceo broj."
                ) from error

            require(
                all(
                    1 <= value <= NUMBER_COUNT
                    for value in numbers
                ),
                f"{path.name}, red {line_number}: broj van opsega 1–39.",
            )
            require(
                len(set(numbers)) == DRAW_SIZE,
                f"{path.name}, red {line_number}: ponovljen broj.",
            )

            rows.append(
                sorted(value - 1 for value in numbers)
            )

    require(
        len(rows) == expected_rows,
        f"{path.name}: očekivano {expected_rows}, pročitano {len(rows)}.",
    )

    draws = np.asarray(rows, dtype=np.int64)

    require(
        draws.shape == (expected_rows, DRAW_SIZE),
        "Pogrešan oblik CSV-a.",
    )

    say(
        "",
        f"ULAZ: {path.name}",
        f"Redova: {len(draws)} | praznih redova: {blank_rows}",
        f"SHA256: {fingerprint}",
        "Redosled kola ostaje redosled iz CSV-a.",
    )

    return draws, fingerprint


def make_context(draws, target_indices, lookback):
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
    feature_count,
):
    size = number_count * 2

    b = (
        parameters[:size].reshape(number_count, 2)
        + 1j
        * parameters[size:2 * size].reshape(number_count, 2)
    )

    w = parameters[2 * size:].reshape(
        feature_count,
        number_count,
    )

    return b, w


def log_amplitudes(x, w):
    t = np.tanh((x @ w) / CAP)
    return CAP * t, 1.0 - t * t


def probability_terms(
    b,
    w,
    x,
    targets,
    space,
    with_gradient=False,
):
    h, link_derivative = log_amplitudes(x, w)
    weights = np.exp(2.0 * h)

    log_z, norm_gradient, occupation = boson_partition(
        b,
        weights,
        space,
        with_gradient=with_gradient,
    )

    amplitude, amplitude_gradient = boson_amplitude(
        b[targets],
        space.v_count,
        with_gradient=with_gradient,
    )

    magnitude = np.abs(amplitude)

    require(
        np.all(magnitude > 0)
        and np.all(np.isfinite(magnitude)),
        "Posmatrano kolo ima nultu ili neispravnu amplitudu.",
    )

    rows = np.arange(len(targets))[:, None]

    log_probability = (
        2.0 * np.log(magnitude)
        + 2.0 * np.sum(h[rows, targets], axis=1)
        - log_z
    )

    require(
        np.all(np.isfinite(log_probability)),
        "Log-verovatnoća nije konačna.",
    )
    require(
        float(np.max(log_probability)) <= 1e-7,
        "Izračunata verovatnoća prelazi 1.",
    )

    return (
        log_probability,
        link_derivative,
        norm_gradient,
        occupation,
        amplitude,
        amplitude_gradient,
    )


def objective(
    parameters,
    x,
    targets,
    ridge,
    number_count,
    space,
):
    b, w = unpack_parameters(
        parameters,
        number_count,
        x.shape[1],
    )

    (
        log_probability,
        link_derivative,
        norm_gradient,
        occupation,
        amplitude,
        amplitude_gradient,
    ) = probability_terms(
        b,
        w,
        x,
        targets,
        space,
        with_gradient=True,
    )

    sample_count = len(targets)

    gradient_b = norm_gradient / sample_count

    selected_gradient = (
        -2.0
        * np.conj(
            amplitude_gradient
            / amplitude[:, None, None]
        )
        / sample_count
    )

    np.add.at(
        gradient_b,
        targets.ravel(),
        selected_gradient.reshape(-1, 2),
    )

    gradient_h = 2.0 * occupation
    rows = np.arange(sample_count)[:, None]
    gradient_h[rows, targets] -= 2.0

    gradient_w = (
        x.T @ (gradient_h * link_derivative)
        / sample_count
        + 2.0 * ridge * w
    )

    # Column scaling is redundant in the normalized distribution.
    # This penalty fixes each column norm without mixing orbitals.
    norm_error = (
        np.sum(np.abs(b) ** 2, axis=0)
        - 1.0
    )

    gauge_penalty = (
        GAUGE_WEIGHT
        * float(np.sum(norm_error * norm_error))
    )

    gradient_b += (
        4.0
        * GAUGE_WEIGHT
        * b
        * norm_error[None, :]
    )

    loss = (
        -float(np.mean(log_probability))
        + ridge * float(np.sum(w * w))
        + gauge_penalty
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

    def orbitals(self, raw_row):
        x = self.transform(
            np.asarray(raw_row).reshape(1, -1)
        )
        h, _ = log_amplitudes(x, self.w)

        return normalize_columns(
            np.exp(h[0])[:, None] * self.b
        )


def fit_model(
    raw_x,
    targets,
    lookback,
    ridge,
    label,
):
    space = make_space(U_COUNT, V_COUNT)

    mean, projection = fit_feature_map(raw_x)
    x = (raw_x - mean) @ projection

    fitted = []

    for phase_order in INITIAL_ORDERS:
        say(
            f"{label}: lag={lookback}, lambda={ridge}, "
            f"početne faze={phase_order}, trening={len(targets)}"
        )

        b0 = initial_orbitals(
            NUMBER_COUNT,
            phase_order,
        )
        w0 = np.zeros(
            (x.shape[1], NUMBER_COUNT),
            dtype=np.float64,
        )

        parameters = pack_parameters(b0, w0)
        iteration = 0

        def callback(current):
            nonlocal iteration
            iteration += 1

            if iteration % 50 == 0:
                say(
                    f"  Optimizacija: {iteration} iteracija"
                )

        def function(current):
            return objective(
                current,
                x,
                targets,
                ridge,
                NUMBER_COUNT,
                space,
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

                say(
                    f"  Početak nije prihvaćen: {message}; "
                    f"max|grad|={last_gradient:.3e}"
                )
                continue

            b, w = unpack_parameters(
                parameters,
                NUMBER_COUNT,
                x.shape[1],
            )

            b = normalize_columns(b)

            train_loss, _ = function(
                pack_parameters(b, w)
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

            say(
                "  Prihvaćen lokalni minimum: "
                f"gubitak={train_loss:.8f}, "
                f"max|grad|={last_gradient:.3e}"
            )

        except (
            RuntimeError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as error:
            say(
                f"  Početak nije prihvaćen: {error}"
            )

    require(
        bool(fitted),
        f"{label}: nijedan deterministički početak "
        "nije prošao numeričke uslove.",
    )

    return min(
        fitted,
        key=lambda model: model.train_loss,
    )


def evaluate_nll(model, raw_x, targets):
    space = make_space(U_COUNT, V_COUNT)
    total = 0.0

    for start in range(0, len(targets), 512):
        stop = min(start + 512, len(targets))

        log_probability = probability_terms(
            model.b,
            model.w,
            model.transform(raw_x[start:stop]),
            targets[start:stop],
            space,
        )[0]

        total -= float(np.sum(log_probability))

    return total / len(targets)


def exhaustive_maximum(
    orbitals,
    space,
    verbose=True,
    label="NEXT",
):
    number_count = len(orbitals)

    expected = math.comb(
        number_count,
        space.draw_size,
    )

    log_z = float(
        boson_partition(
            orbitals,
            np.ones((1, number_count)),
            space,
        )[0][0]
    )

    iterator = combinations(
        range(number_count),
        space.draw_size,
    )

    visited = 0
    best_combination = None
    best_log_probability = -math.inf
    log_total_probability = -math.inf
    previous_progress = 0

    while True:
        block = list(islice(iterator, MAP_BATCH))

        if not block:
            break

        indices = np.asarray(
            block,
            dtype=np.int64,
        )

        amplitude, _ = boson_amplitude(
            orbitals[indices],
            space.v_count,
        )

        # Zero amplitudes are valid during exhaustive enumeration.
        with np.errstate(divide="ignore"):
            scores = (
                2.0 * np.log(np.abs(amplitude))
                - log_z
            )

        require(
            not np.any(np.isnan(scores))
            and not np.any(np.isposinf(scores)),
            "Neispravna verovatnoća tokom završne pretrage.",
        )

        local_index = int(np.argmax(scores))
        local_score = float(scores[local_index])

        if local_score > best_log_probability:
            best_log_probability = local_score
            best_combination = tuple(
                int(value)
                for value in indices[local_index]
            )

        log_total_probability = float(
            np.logaddexp(
                log_total_probability,
                logsumexp(scores),
            )
        )

        visited += len(indices)
        progress = visited // 2_000_000

        if verbose and progress > previous_progress:
            say(
                f"{label}: pregledano "
                f"{visited:,}/{expected:,}"
            )
            previous_progress = progress

    require(
        visited == expected,
        "Završna pretraga nije potpuna.",
    )
    require(
        best_combination is not None
        and np.isfinite(best_log_probability),
        "Nije pronađen konačan maksimum raspodele.",
    )
    require(
        best_log_probability <= 1e-7,
        "Maksimalna verovatnoća prelazi 1.",
    )

    probability_sum = math.exp(
        log_total_probability
    )

    require(
        abs(probability_sum - 1.0)
        <= NORMALIZATION_TOLERANCE,
        f"Normalizacija nije prošla: zbir={probability_sum:.12g}.",
    )

    return (
        best_combination,
        best_log_probability,
        probability_sum,
        visited,
    )


def self_checks():
    say(
        "",
        "Pokrećem matematičke i numeričke samoprovere."
    )

    number_count = 9
    b = initial_orbitals(
        number_count,
        phase_order=2,
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

    w = 0.025 * np.sin(
        np.arange(2 * number_count).reshape(
            2,
            number_count,
        )
        + 0.3
    )

    for u_count, v_count in ((2, 1), (4, 3)):
        space = make_space(u_count, v_count)

        all_sets = np.asarray(
            list(
                combinations(
                    range(number_count),
                    space.draw_size,
                )
            ),
            dtype=np.int64,
        )

        target_positions = np.array(
            [
                0,
                len(all_sets) // 3,
                len(all_sets) // 2,
                len(all_sets) - 1,
            ],
            dtype=np.int64,
        )
        targets = all_sets[target_positions]

        parameters = pack_parameters(b, w)

        _, analytic = objective(
            parameters,
            x,
            targets,
            0.03,
            number_count,
            space,
        )

        numerical = np.empty_like(parameters)
        epsilon = 1e-6

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
                space,
            )[0]

            f_minus = objective(
                minus,
                x,
                targets,
                0.03,
                number_count,
                space,
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
            f"Izvod za zauzeće {u_count}+{v_count} "
            f"nije prošao: {gradient_error:.3e}.",
        )

        repeated_x = np.repeat(
            x[:1],
            len(all_sets),
            axis=0,
        )

        log_probability = probability_terms(
            b,
            w,
            repeated_x,
            all_sets,
            space,
        )[0]

        probability_sum = float(
            np.exp(logsumexp(log_probability))
        )

        require(
            abs(probability_sum - 1.0) < 1e-9,
            "Potpuna mala provera normalizacije nije prošla.",
        )

        selected = b[targets[:1]]

        amplitude, _ = boson_amplitude(
            selected,
            v_count,
        )
        reversed_amplitude, _ = boson_amplitude(
            selected[:, ::-1, :],
            v_count,
        )

        require(
            np.allclose(
                amplitude,
                reversed_amplitude,
                rtol=1e-11,
                atol=1e-13,
            ),
            "Bozonska simetrija nije prošla proveru.",
        )

        # Independent permanent computation on one small matrix.
        column_indices = (
            [0] * u_count
            + [1] * v_count
        )
        permanent_matrix = (
            selected[0][:, column_indices]
        )

        direct_permanent = 0.0j

        for permutation in permutations(
            range(space.draw_size)
        ):
            term = 1.0 + 0.0j

            for row, column in enumerate(permutation):
                term *= permanent_matrix[row, column]

            direct_permanent += term

        expected_permanent = (
            math.factorial(u_count)
            * math.factorial(v_count)
            * amplitude[0]
        )

        require(
            np.allclose(
                direct_permanent,
                expected_permanent,
                rtol=1e-10,
                atol=1e-12,
            ),
            "Veza između amplitude i permanenta nije prošla.",
        )

        # Bosons do not have Pauli exclusion at the primitive level.
        repeated_rows = np.repeat(
            b[0][None, None, :],
            space.draw_size,
            axis=1,
        )

        repeated_amplitude, _ = boson_amplitude(
            repeated_rows,
            v_count,
        )

        expected_repeated = (
            math.comb(space.draw_size, v_count)
            * b[0, 0] ** u_count
            * b[0, 1] ** v_count
        )

        require(
            np.allclose(
                repeated_amplitude[0],
                expected_repeated,
                rtol=1e-11,
                atol=1e-13,
            )
            and abs(repeated_amplitude[0]) > 0,
            "Provera ponovljenog bozonskog stanja nije prošla.",
        )

        h, _ = log_amplitudes(x[:1], w)

        orbitals = normalize_columns(
            np.exp(h[0])[:, None] * b
        )

        best, best_log_probability, _, _ = exhaustive_maximum(
            orbitals,
            space,
            verbose=False,
        )

        require(
            abs(
                best_log_probability
                - float(np.max(log_probability))
            ) < 1e-9,
            "Provera maksimuma nije prošla.",
        )

        matches = np.flatnonzero(
            np.all(
                all_sets == np.asarray(best),
                axis=1,
            )
        )

        require(
            len(matches) == 1
            and abs(
                best_log_probability
                - float(log_probability[matches[0]])
            ) < 1e-9,
            "Maksimum i vraćena kombinacija nisu saglasni.",
        )

        say(
            f"Samoprovera {u_count}+{v_count}: "
            f"stanja rekurzije={space.size}, "
            f"greška izvoda={gradient_error:.3e}, "
            f"zbir verovatnoća={probability_sum:.12f}"
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
        context[0, 39] == 1.0
        and context[0, 79] == 0.0
        and context[0, 119] == 0.0
        and float(context.sum()) == 8.0,
        "Provera početnog istorijskog prelaza nije prošla.",
    )

    say(
        "Samoprovere prošle: izvod, normalizacija, "
        "simetrija, permanent, bozonsko zauzeće, "
        "maksimum i početni prelaz."
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
            longest = max(longest, current)
        else:
            current = 1

    return longest


def context_sensitivity(
    model,
    raw,
    raw_next,
    draws,
    space,
):
    # Compare actual model probabilities on fixed historical probes.
    # This diagnostic does not change the prediction.
    probes = np.unique(
        draws[-min(16, len(draws)):],
        axis=0,
    )

    next_inputs = np.repeat(
        raw_next[None, :],
        len(probes),
        axis=0,
    )

    next_log_probability = probability_terms(
        model.b,
        model.w,
        model.transform(next_inputs),
        probes,
        space,
    )[0]

    sensitivity = 0.0

    for raw_previous in raw[-8:]:
        previous_inputs = np.repeat(
            raw_previous[None, :],
            len(probes),
            axis=0,
        )

        previous_log_probability = probability_terms(
            model.b,
            model.w,
            model.transform(previous_inputs),
            probes,
            space,
        )[0]

        sensitivity = max(
            sensitivity,
            float(
                np.max(
                    np.abs(
                        next_log_probability
                        - previous_log_probability
                    )
                )
            ),
        )

    return sensitivity


def process_file(path, draws, fingerprint):
    space = make_space(U_COUNT, V_COUNT)

    targets = draws[1:].copy()
    sample_count = len(targets)

    train_60 = int(0.60 * sample_count)
    train_70 = int(0.70 * sample_count)
    train_80 = int(0.80 * sample_count)

    require(
        50 < train_60 < train_70 < train_80 < sample_count,
        "Premalo podataka za hronološku podelu.",
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
        ) in enumerate(folds, start=1):
            model = fit_model(
                raw[:train_end],
                targets[:train_end],
                lookback,
                ridge,
                label=(
                    f"{path.name} / validacija {fold_number}"
                ),
            )

            count = validation_end - train_end

            nll = evaluate_nll(
                model,
                raw[train_end:validation_end],
                targets[train_end:validation_end],
            )

            total_nll += count * nll
            total_count += count

            say(
                f"  Validacioni NLL po kolu: {nll:.8f}"
            )

        score = total_nll / total_count
        scores.append((score, lookback, ridge))

        say(
            f"KANDIDAT: lag={lookback}, lambda={ridge}, "
            f"zbirni validacioni NLL={score:.8f}"
        )

    validation_nll, selected_lookback, selected_ridge = min(
        scores,
        key=lambda item: item[0],
    )

    raw = contexts[selected_lookback]

    say(
        "",
        f"IZABRANO: lag={selected_lookback}, "
        f"lambda={selected_ridge}, "
        f"validacioni NLL={validation_nll:.8f}",
    )

    test_model = fit_model(
        raw[:train_80],
        targets[:train_80],
        selected_lookback,
        selected_ridge,
        label=f"{path.name} / odvojeni test",
    )

    test_nll = evaluate_nll(
        test_model,
        raw[train_80:],
        targets[train_80:],
    )

    say(
        f"ODVOJENI TEST: {sample_count - train_80} kola, "
        f"NLL po kolu={test_nll:.8f}",
        "Test se ne koristi za promenu parametara.",
    )

    final_model = fit_model(
        raw,
        targets,
        selected_lookback,
        selected_ridge,
        label=f"{path.name} / završno učenje",
    )

    say(
        f"ZAVRŠNO UČENJE: svih {sample_count} prelaza "
        f"iz {len(draws)} kola."
    )

    raw_next = make_context(
        draws,
        [len(draws)],
        selected_lookback,
    )[0]

    sensitivity = context_sensitivity(
        final_model,
        raw,
        raw_next,
        draws,
        space,
    )

    orbitals = final_model.orbitals(raw_next)

    best, log_probability, probability_sum, visited = (
        exhaustive_maximum(
            orbitals,
            space,
            verbose=True,
            label=path.name,
        )
    )

    require(
        len(best) == DRAW_SIZE
        and len(set(best)) == DRAW_SIZE
        and all(
            0 <= value < NUMBER_COUNT
            for value in best
        ),
        "NEXT nije legalna kombinacija 7/39.",
    )

    run_length = longest_consecutive_run(best)

    historical_repetitions = int(
        np.count_nonzero(
            np.all(
                draws == np.asarray(best)[None, :],
                axis=1,
            )
        )
    )

    warnings = []

    if run_length >= 5:
        warnings.append(
            f"NEXT sadrži niz od {run_length} uzastopnih brojeva. "
            "Potreban je pregled modela. Rezultat nije filtriran."
        )

    if historical_repetitions:
        warnings.append(
            f"NEXT se pojavljuje {historical_repetitions} puta "
            "u istoriji. Rezultat nije promenjen."
        )

    if sensitivity < 1e-8:
        warnings.append(
            "Na proverenim istorijskim kombinacijama nije "
            "utvrđen numerički uticaj promene konteksta. "
            "Potrebna je provera uticaja istorijskog ulaza."
        )

    return {
        "name": path.name,
        "fingerprint": fingerprint,
        "combination": tuple(value + 1 for value in best),
        "validation_nll": validation_nll,
        "test_nll": test_nll,
        "model_probability": math.exp(log_probability),
        "probability_sum": probability_sum,
        "visited": visited,
        "sensitivity": sensitivity,
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

    require(
        U_COUNT + V_COUNT == DRAW_SIZE,
        "Zbir orbitalnih zauzeća mora biti sedam.",
    )

    space = make_space(U_COUNT, V_COUNT)

    say(
        "V2 BOSON — Loto 7/39",
        "Lokalni proračun simetričnih kompleksnih amplituda.",
        f"Orbitalna zauzeća: {U_COUNT}+{V_COUNT}",
        f"Broj stanja normalizacione rekurzije: {space.size}",
        "Bez slučajnog generisanja i bez klasifikatora.",
        "Učenje i završna pretraga mogu potrajati.",
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
            (path, draws, fingerprint)
        )

    require(
        len(
            {
                fingerprint
                for _, _, fingerprint in datasets
            }
        ) == 3,
        "Dva CSV-a imaju identičan sadržaj.",
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
        for right in range(left + 1, len(results)):
            if (
                results[left]["combination"]
                == results[right]["combination"]
            ):
                message = (
                    "Ista NEXT kombinacija za različite CSV-ove: "
                    f"{results[left]['name']} i "
                    f"{results[right]['name']}. "
                    "Proveriti uticaj istorije i moguću degeneraciju. "
                    "Kombinacija nije promenjena."
                )

                results[left]["warnings"].append(message)
                results[right]["warnings"].append(message)

    say(
        "",
        "============================================================",
        "ZAVRŠNI REZULTATI V2 BOSON",
        "============================================================",
        "Modelna verovatnoća pripada naučenom modelu.",
        "Ona nije potvrđena verovatnoća stvarnog narednog izvlačenja.",
        "NLL koristi prirodni logaritam. Niži NLL je bolji na tom skupu.",
    )

    for result in results:
        numbers = " ".join(
            f"{value:02d}"
            for value in result["combination"]
        )

        say(
            "",
            f"CSV: {result['name']}",
            f"NEXT: {numbers}",
            f"Validacioni NLL: {result['validation_nll']:.8f}",
            f"Odvojeni test NLL: {result['test_nll']:.8f}",
            "Modelna verovatnoća NEXT: "
            f"{result['model_probability']:.12g}",
            f"Pregledano kombinacija: {result['visited']:,}",
            "Zbir svih verovatnoća: "
            f"{result['probability_sum']:.12f}",
            "Najveća promena log-verovatnoće u proveri konteksta: "
            f"{result['sensitivity']:.6e}",
            f"Najduži uzastopni niz: {result['run_length']}",
            f"Raspon kombinacije: {result['span']}",
        )

        for warning in result["warnings"]:
            say(
                f"PROVERA MODELA: {warning}"
            )

    say(
        "",
        "Završna pretraga je potpuna za svaki prihvaćeni model.",
        "Učenje je lokalna numerička optimizacija.",
        "Globalni optimum parametara nije dokazan.",
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
            f"ZAUSTAVLJENO: {error}",
            file=sys.stderr,
            flush=True,
        )
        print(
            "Završni skup od tri NEXT rezultata nije potvrđen.",
            file=sys.stderr,
            flush=True,
        )
        print(
            "Tačno mesto prekida:",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        sys.exit(1)



"""
V2 BOSON — Loto 7/39
Lokalni proračun simetričnih kompleksnih amplituda.
Orbitalna zauzeća: 4+3
Broj stanja normalizacione rekurzije: 60
Bez slučajnog generisanja i bez klasifikatora.
Učenje i završna pretraga mogu potrajati.

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
Samoprovera 2+1: stanja rekurzije=10, greška izvoda=5.896e-10, zbir verovatnoća=1.000000000000
Samoprovera 4+3: stanja rekurzije=60, greška izvoda=6.597e-10, zbir verovatnoća=1.000000000000
Samoprovere prošle: izvod, normalizacija, simetrija, permanent, bozonsko zauzeće, maksimum i početni prelaz.
loto7_4710_k86.csv / validacija 1: lag=1, lambda=0.03, početne faze=17, trening=2825
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.46425344, max|grad|=8.564e-07
loto7_4710_k86.csv / validacija 1: lag=1, lambda=0.03, početne faze=11, trening=2825
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.46672594, max|grad|=5.238e-06
  Validacioni NLL po kolu: 16.64747075
loto7_4710_k86.csv / validacija 2: lag=1, lambda=0.03, početne faze=17, trening=3296
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.47293458, max|grad|=2.133e-06
loto7_4710_k86.csv / validacija 2: lag=1, lambda=0.03, početne faze=11, trening=3296
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Prihvaćen lokalni minimum: gubitak=16.47493317, max|grad|=9.250e-07
  Validacioni NLL po kolu: 16.58205502
KANDIDAT: lag=1, lambda=0.03, zbirni validacioni NLL=16.61476288
loto7_4710_k86.csv / validacija 1: lag=1, lambda=0.3, početne faze=17, trening=2825
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.48637740, max|grad|=8.673e-07
loto7_4710_k86.csv / validacija 1: lag=1, lambda=0.3, početne faze=11, trening=2825
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Prihvaćen lokalni minimum: gubitak=16.48723837, max|grad|=5.495e-07
  Validacioni NLL po kolu: 16.62336296
loto7_4710_k86.csv / validacija 2: lag=1, lambda=0.3, početne faze=17, trening=3296
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Optimizacija: 300 iteracija
  Optimizacija: 350 iteracija
  Prihvaćen lokalni minimum: gubitak=16.49562648, max|grad|=7.640e-07
loto7_4710_k86.csv / validacija 2: lag=1, lambda=0.3, početne faze=11, trening=3296
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Prihvaćen lokalni minimum: gubitak=16.49510401, max|grad|=7.198e-07
  Validacioni NLL po kolu: 16.57911287
KANDIDAT: lag=1, lambda=0.3, zbirni validacioni NLL=16.60123791
loto7_4710_k86.csv / validacija 1: lag=3, lambda=0.03, početne faze=17, trening=2825
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.45952645, max|grad|=9.603e-07
loto7_4710_k86.csv / validacija 1: lag=3, lambda=0.03, početne faze=11, trening=2825
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.46205922, max|grad|=7.729e-07
  Validacioni NLL po kolu: 16.65304024
loto7_4710_k86.csv / validacija 2: lag=3, lambda=0.03, početne faze=17, trening=3296
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.47354666, max|grad|=9.525e-07
loto7_4710_k86.csv / validacija 2: lag=3, lambda=0.03, početne faze=11, trening=3296
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.47316559, max|grad|=9.875e-07
  Validacioni NLL po kolu: 16.60541086
KANDIDAT: lag=3, lambda=0.03, zbirni validacioni NLL=16.62922555
loto7_4710_k86.csv / validacija 1: lag=3, lambda=0.3, početne faze=17, trening=2825
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.48378195, max|grad|=9.749e-07
loto7_4710_k86.csv / validacija 1: lag=3, lambda=0.3, početne faze=11, trening=2825
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Prihvaćen lokalni minimum: gubitak=16.48608882, max|grad|=9.923e-07
  Validacioni NLL po kolu: 16.62673421
loto7_4710_k86.csv / validacija 2: lag=3, lambda=0.3, početne faze=17, trening=3296
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.49460042, max|grad|=8.594e-07
loto7_4710_k86.csv / validacija 2: lag=3, lambda=0.3, početne faze=11, trening=3296
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Prihvaćen lokalni minimum: gubitak=16.49409597, max|grad|=8.395e-07
  Validacioni NLL po kolu: 16.58387701
KANDIDAT: lag=3, lambda=0.3, zbirni validacioni NLL=16.60530561

IZABRANO: lag=1, lambda=0.3, validacioni NLL=16.60123791
loto7_4710_k86.csv / odvojeni test: lag=1, lambda=0.3, početne faze=17, trening=3767
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Prihvaćen lokalni minimum: gubitak=16.50499916, max|grad|=7.004e-07
loto7_4710_k86.csv / odvojeni test: lag=1, lambda=0.3, početne faze=11, trening=3767
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.49908878, max|grad|=9.243e-07
ODVOJENI TEST: 942 kola, NLL po kolu=16.57949386
Test se ne koristi za promenu parametara.
loto7_4710_k86.csv / završno učenje: lag=1, lambda=0.3, početne faze=17, trening=4709
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Optimizacija: 300 iteracija
  Optimizacija: 350 iteracija
  Prihvaćen lokalni minimum: gubitak=16.50725380, max|grad|=8.514e-07
loto7_4710_k86.csv / završno učenje: lag=1, lambda=0.3, početne faze=11, trening=4709
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Prihvaćen lokalni minimum: gubitak=16.50878640, max|grad|=8.424e-07
ZAVRŠNO UČENJE: svih 4709 prelaza iz 4710 kola.
loto7_4710_k86.csv: pregledano 2,002,944/15,380,937
loto7_4710_k86.csv: pregledano 4,001,792/15,380,937
loto7_4710_k86.csv: pregledano 6,000,640/15,380,937
loto7_4710_k86.csv: pregledano 8,003,584/15,380,937
loto7_4710_k86.csv: pregledano 10,002,432/15,380,937
loto7_4710_k86.csv: pregledano 12,001,280/15,380,937
loto7_4710_k86.csv: pregledano 14,000,128/15,380,937
loto7_4710_k86_loto_2977.csv / validacija 1: lag=1, lambda=0.03, početne faze=17, trening=1785
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Prihvaćen lokalni minimum: gubitak=16.41138979, max|grad|=9.389e-07
loto7_4710_k86_loto_2977.csv / validacija 1: lag=1, lambda=0.03, početne faze=11, trening=1785
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.41345489, max|grad|=1.176e-06
  Validacioni NLL po kolu: 16.64123902
loto7_4710_k86_loto_2977.csv / validacija 2: lag=1, lambda=0.03, početne faze=17, trening=2083
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.42732276, max|grad|=8.509e-07
loto7_4710_k86_loto_2977.csv / validacija 2: lag=1, lambda=0.03, početne faze=11, trening=2083
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Prihvaćen lokalni minimum: gubitak=16.43082186, max|grad|=8.437e-07
  Validacioni NLL po kolu: 16.62696995
KANDIDAT: lag=1, lambda=0.03, zbirni validacioni NLL=16.63411648
loto7_4710_k86_loto_2977.csv / validacija 1: lag=1, lambda=0.3, početne faze=17, trening=1785
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.44562605, max|grad|=9.312e-07
loto7_4710_k86_loto_2977.csv / validacija 1: lag=1, lambda=0.3, početne faze=11, trening=1785
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.45039582, max|grad|=9.838e-07
  Validacioni NLL po kolu: 16.60050596
loto7_4710_k86_loto_2977.csv / validacija 2: lag=1, lambda=0.3, početne faze=17, trening=2083
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Prihvaćen lokalni minimum: gubitak=16.46016763, max|grad|=8.035e-07
loto7_4710_k86_loto_2977.csv / validacija 2: lag=1, lambda=0.3, početne faze=11, trening=2083
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Optimizacija: 300 iteracija
  Prihvaćen lokalni minimum: gubitak=16.46344006, max|grad|=6.856e-07
  Validacioni NLL po kolu: 16.60408913
KANDIDAT: lag=1, lambda=0.3, zbirni validacioni NLL=16.60229453
loto7_4710_k86_loto_2977.csv / validacija 1: lag=3, lambda=0.03, početne faze=17, trening=1785
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Prihvaćen lokalni minimum: gubitak=16.42004850, max|grad|=7.865e-07
loto7_4710_k86_loto_2977.csv / validacija 1: lag=3, lambda=0.03, početne faze=11, trening=1785
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Optimizacija: 300 iteracija
  Optimizacija: 350 iteracija
  Prihvaćen lokalni minimum: gubitak=16.41328759, max|grad|=6.048e-07
  Validacioni NLL po kolu: 16.66816046
loto7_4710_k86_loto_2977.csv / validacija 2: lag=3, lambda=0.03, početne faze=17, trening=2083
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Prihvaćen lokalni minimum: gubitak=16.42521514, max|grad|=6.465e-07
loto7_4710_k86_loto_2977.csv / validacija 2: lag=3, lambda=0.03, početne faze=11, trening=2083
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Prihvaćen lokalni minimum: gubitak=16.42818536, max|grad|=2.195e-06
  Validacioni NLL po kolu: 16.58830048
KANDIDAT: lag=3, lambda=0.03, zbirni validacioni NLL=16.62829758
loto7_4710_k86_loto_2977.csv / validacija 1: lag=3, lambda=0.3, početne faze=17, trening=1785
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.44550640, max|grad|=8.071e-07
loto7_4710_k86_loto_2977.csv / validacija 1: lag=3, lambda=0.3, početne faze=11, trening=1785
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.45169752, max|grad|=8.751e-07
  Validacioni NLL po kolu: 16.61446571
loto7_4710_k86_loto_2977.csv / validacija 2: lag=3, lambda=0.3, početne faze=17, trening=2083
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Prihvaćen lokalni minimum: gubitak=16.45872718, max|grad|=7.805e-07
loto7_4710_k86_loto_2977.csv / validacija 2: lag=3, lambda=0.3, početne faze=11, trening=2083
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Prihvaćen lokalni minimum: gubitak=16.46195494, max|grad|=6.075e-07
  Validacioni NLL po kolu: 16.57795187
KANDIDAT: lag=3, lambda=0.3, zbirni validacioni NLL=16.59623947

IZABRANO: lag=3, lambda=0.3, validacioni NLL=16.59623947
loto7_4710_k86_loto_2977.csv / odvojeni test: lag=3, lambda=0.3, početne faze=17, trening=2380
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.46477853, max|grad|=9.666e-07
loto7_4710_k86_loto_2977.csv / odvojeni test: lag=3, lambda=0.3, početne faze=11, trening=2380
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Prihvaćen lokalni minimum: gubitak=16.46828453, max|grad|=3.099e-06
ODVOJENI TEST: 596 kola, NLL po kolu=16.59584649
Test se ne koristi za promenu parametara.
loto7_4710_k86_loto_2977.csv / završno učenje: lag=3, lambda=0.3, početne faze=17, trening=2976
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Optimizacija: 300 iteracija
  Optimizacija: 350 iteracija
  Optimizacija: 400 iteracija
  Prihvaćen lokalni minimum: gubitak=16.47943959, max|grad|=7.937e-07
loto7_4710_k86_loto_2977.csv / završno učenje: lag=3, lambda=0.3, početne faze=11, trening=2976
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Prihvaćen lokalni minimum: gubitak=16.48020712, max|grad|=9.005e-07
ZAVRŠNO UČENJE: svih 2976 prelaza iz 2977 kola.
loto7_4710_k86_loto_2977.csv: pregledano 2,002,944/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 4,001,792/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 6,000,640/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 8,003,584/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 10,002,432/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 12,001,280/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 14,000,128/15,380,937
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=1, lambda=0.03, početne faze=17, trening=1039
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Prihvaćen lokalni minimum: gubitak=16.31499832, max|grad|=6.820e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=1, lambda=0.03, početne faze=11, trening=1039
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Prihvaćen lokalni minimum: gubitak=16.32013989, max|grad|=7.999e-07
  Validacioni NLL po kolu: 16.77379873
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=1, lambda=0.03, početne faze=17, trening=1212
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Prihvaćen lokalni minimum: gubitak=16.35284373, max|grad|=1.066e-06
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=1, lambda=0.03, početne faze=11, trening=1212
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Prihvaćen lokalni minimum: gubitak=16.35824450, max|grad|=8.449e-07
  Validacioni NLL po kolu: 16.75975685
KANDIDAT: lag=1, lambda=0.03, zbirni validacioni NLL=16.76677779
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=1, lambda=0.3, početne faze=17, trening=1039
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.37559521, max|grad|=8.479e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=1, lambda=0.3, početne faze=11, trening=1039
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Prihvaćen lokalni minimum: gubitak=16.38552995, max|grad|=7.813e-07
  Validacioni NLL po kolu: 16.68153168
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=1, lambda=0.3, početne faze=17, trening=1212
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Prihvaćen lokalni minimum: gubitak=16.40177707, max|grad|=7.259e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=1, lambda=0.3, početne faze=11, trening=1212
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Prihvaćen lokalni minimum: gubitak=16.40148422, max|grad|=1.264e-06
  Validacioni NLL po kolu: 16.66405536
KANDIDAT: lag=1, lambda=0.3, zbirni validacioni NLL=16.67279352
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=3, lambda=0.03, početne faze=17, trening=1039
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Prihvaćen lokalni minimum: gubitak=16.30873356, max|grad|=9.886e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=3, lambda=0.03, početne faze=11, trening=1039
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.31679695, max|grad|=6.661e-07
  Validacioni NLL po kolu: 16.68764178
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=3, lambda=0.03, početne faze=17, trening=1212
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Prihvaćen lokalni minimum: gubitak=16.34199918, max|grad|=7.235e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=3, lambda=0.03, početne faze=11, trening=1212
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Prihvaćen lokalni minimum: gubitak=16.34639431, max|grad|=9.314e-07
  Validacioni NLL po kolu: 16.74964620
KANDIDAT: lag=3, lambda=0.03, zbirni validacioni NLL=16.71864399
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=3, lambda=0.3, početne faze=17, trening=1039
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.37208849, max|grad|=9.109e-07
loto7_4710_k86_loto_plus_1733.csv / validacija 1: lag=3, lambda=0.3, početne faze=11, trening=1039
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.36785989, max|grad|=8.069e-07
  Validacioni NLL po kolu: 16.70262723
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=3, lambda=0.3, početne faze=17, trening=1212
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Optimizacija: 250 iteracija
  Prihvaćen lokalni minimum: gubitak=16.38733522, max|grad|=2.526e-06
loto7_4710_k86_loto_plus_1733.csv / validacija 2: lag=3, lambda=0.3, početne faze=11, trening=1212
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Prihvaćen lokalni minimum: gubitak=16.39312105, max|grad|=9.286e-07
  Validacioni NLL po kolu: 16.71829869
KANDIDAT: lag=3, lambda=0.3, zbirni validacioni NLL=16.71046296

IZABRANO: lag=1, lambda=0.3, validacioni NLL=16.67279352
loto7_4710_k86_loto_plus_1733.csv / odvojeni test: lag=1, lambda=0.3, početne faze=17, trening=1385
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Prihvaćen lokalni minimum: gubitak=16.42329584, max|grad|=6.126e-07
loto7_4710_k86_loto_plus_1733.csv / odvojeni test: lag=1, lambda=0.3, početne faze=11, trening=1385
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.42247773, max|grad|=9.905e-07
ODVOJENI TEST: 347 kola, NLL po kolu=16.65515856
Test se ne koristi za promenu parametara.
loto7_4710_k86_loto_plus_1733.csv / završno učenje: lag=1, lambda=0.3, početne faze=17, trening=1732
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Optimizacija: 200 iteracija
  Prihvaćen lokalni minimum: gubitak=16.44940230, max|grad|=7.485e-07
loto7_4710_k86_loto_plus_1733.csv / završno učenje: lag=1, lambda=0.3, početne faze=11, trening=1732
  Optimizacija: 50 iteracija
  Optimizacija: 100 iteracija
  Optimizacija: 150 iteracija
  Prihvaćen lokalni minimum: gubitak=16.44121970, max|grad|=8.453e-07
ZAVRŠNO UČENJE: svih 1732 prelaza iz 1733 kola.
loto7_4710_k86_loto_plus_1733.csv: pregledano 2,002,944/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 4,001,792/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 6,000,640/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 8,003,584/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 10,002,432/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 12,001,280/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 14,000,128/15,380,937



============================================================
ZAVRŠNI REZULTATI V2 BOSON
============================================================
Modelna verovatnoća pripada naučenom modelu.
Ona nije potvrđena verovatnoća stvarnog narednog izvlačenja.
NLL koristi prirodni logaritam. Niži NLL je bolji na tom skupu.

CSV: loto7_4710_k86.csv
NEXT: 03 05 06 09 21 23 35
Validacioni NLL: 16.60123791
Odvojeni test NLL: 16.57949386
Modelna verovatnoća NEXT: 1.01451150237e-06
Pregledano kombinacija: 15,380,937
Zbir svih verovatnoća: 1.000000000000
Najveća promena log-verovatnoće u proveri konteksta: 6.632904e-01
Najduži uzastopni niz: 2
Raspon kombinacije: 32

CSV: loto7_4710_k86_loto_2977.csv
NEXT: 07 18 21 26 35 37 38
Validacioni NLL: 16.59623947
Odvojeni test NLL: 16.59584649
Modelna verovatnoća NEXT: 6.17403495343e-07
Pregledano kombinacija: 15,380,937
Zbir svih verovatnoća: 1.000000000000
Najveća promena log-verovatnoće u proveri konteksta: 7.197562e-01
Najduži uzastopni niz: 2
Raspon kombinacije: 31

CSV: loto7_4710_k86_loto_plus_1733.csv
NEXT: 01 05 06 09 13 18 38
Validacioni NLL: 16.67279352
Odvojeni test NLL: 16.65515856
Modelna verovatnoća NEXT: 1.95861679357e-06
Pregledano kombinacija: 15,380,937
Zbir svih verovatnoća: 1.000000000000
Najveća promena log-verovatnoće u proveri konteksta: 8.496409e-01
Najduži uzastopni niz: 2
Raspon kombinacije: 37

Završna pretraga je potpuna za svaki prihvaćeni model.
Učenje je lokalna numerička optimizacija.
Globalni optimum parametara nije dokazan.
"""



"""
Bozoni imaju simetrične amplitude i dopuštaju višestruko zauzimanje istog stanja. 
Za Loto 7/39 model ograničiti na sedam različitih brojeva i posebno proveriti normalizaciju.

Bozonski model - simetrične amplitude, uz ograničenje izlaza na sedam različitih brojeva. 

V2 koristi simetrične bozonske amplitude sa dve naučene orbitale i zauzećima 4+3. 
Amplituda je proporcionalna permanentu; raspodela se normalizuje nad kombinacijama sa sedam različitih brojeva. 
Adaptacija za Loto 7/39 - osnova bozonskih amplituda. 

Rekurzija normalizacije i njen izvod u odvojenom malom numeričkom primeru.
"""
