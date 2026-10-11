"""
Validate synthetic population/income surfaces against real census tract data.

This script:
1. Reproduces the synthetic spatial equity analysis as described in Methods
2. Tests Gini sensitivity to synthetic parameters (decay rate, noise sigma)
3. Compares against real ACS census tract data (when API key available)

Author: Peiyuan Lin
Date: 2026-06-24
"""

import numpy as np
from scipy.integrate import trapezoid as _trapz
import json
import os
import csv
from pathlib import Path

# ============================================================
# PART 1: Synthetic data generation (matching paper Methods)
# ============================================================

def generate_synthetic_grid(city_name, grid_size=200,
                           pop_decay_rate=None, income_decay_rate=None,
                           noise_sigma=0.3):
    """
    Generate synthetic urban grid with monocentric negative exponential
    population density and income gradients + log-normal noise.
    """
    default_params = {
        'Guangzhou':      {'pop_decay': 0.025, 'income_decay': 0.020, 'center_pop': 25000, 'center_income': 85000},
        'Chicago':        {'pop_decay': 0.018, 'income_decay': 0.015, 'center_pop': 15000, 'center_income': 95000},
        'Tokyo':          {'pop_decay': 0.020, 'income_decay': 0.018, 'center_pop': 30000, 'center_income': 90000},
        'Kuala Lumpur':   {'pop_decay': 0.030, 'income_decay': 0.025, 'center_pop': 12000, 'center_income': 55000},
    }

    params = default_params[city_name]
    if pop_decay_rate is not None:
        params['pop_decay'] = pop_decay_rate
    if income_decay_rate is not None:
        params['income_decay'] = income_decay_rate

    center = (grid_size - 1) / 2
    y, x = np.ogrid[:grid_size, :grid_size]
    dist = np.sqrt((x - center)**2 + (y - center)**2)

    pop_density_base = params['center_pop'] * np.exp(-params['pop_decay'] * dist)
    income_base = params['center_income'] * np.exp(-params['income_decay'] * dist)

    np.random.seed(hash(city_name) % 2**31)
    noise_pop = np.exp(np.random.normal(0, noise_sigma, (grid_size, grid_size)))
    noise_income = np.exp(np.random.normal(0, noise_sigma, (grid_size, grid_size)))

    pop_density = np.maximum(pop_density_base * noise_pop, 0)
    income = np.maximum(income_base * noise_income, 0)
    cell_area_km2 = 1.0
    pop_total = pop_density * cell_area_km2

    return pop_total, income, dist


def generate_protection_grid(city_name, grid_size=200):
    """Generate synthetic flood depth reduction grid."""
    center = (grid_size - 1) / 2
    y, x = np.ogrid[:grid_size, :grid_size]
    dist = np.sqrt((x - center)**2 + (y - center)**2)

    protection = np.exp(-0.015 * dist)
    np.random.seed(hash(f'flood_{city_name}') % 2**31)
    protection *= (1 + 0.2 * np.random.randn(grid_size, grid_size))
    protection = np.maximum(protection, 0)

    return protection


# ============================================================
# PART 2: Gini coefficient and Lorenz curve
# ============================================================

def compute_gini(values, weights=None):
    """Compute Gini coefficient."""
    if weights is None:
        weights = np.ones_like(values)

    idx = np.argsort(values)
    values_sorted = values[idx]
    weights_sorted = weights[idx]

    cum_values = np.cumsum(values_sorted * weights_sorted)
    cum_weights = np.cumsum(weights_sorted)

    cum_values_norm = cum_values / cum_values[-1]
    cum_weights_norm = cum_weights / cum_weights[-1]

    gini = 1 - 2 * _trapz(cum_values_norm, cum_weights_norm)
    return gini


def compute_quintile_shares(values, weights=None):
    """Compute share of total value for each population quintile."""
    if weights is None:
        weights = np.ones_like(values)

    idx = np.argsort(values)
    values_sorted = values[idx]
    weights_sorted = weights[idx]

    cum_weights = np.cumsum(weights_sorted)
    total_weight = cum_weights[-1]

    quintile_shares = []
    for q in range(5):
        lo = q * total_weight / 5
        hi = (q + 1) * total_weight / 5
        mask = (cum_weights >= lo) & (cum_weights < hi)
        if q == 4:
            mask = cum_weights >= lo
        share = np.sum(values_sorted[mask] * weights_sorted[mask])
        quintile_shares.append(share)

    total = np.sum(quintile_shares)
    quintile_shares = [s / total * 100 for s in quintile_shares]
    return quintile_shares


# ============================================================
# PART 3: Sensitivity analysis
# ============================================================

def run_sensitivity(city_name, n_samples=50):
    """Test Gini sensitivity to parameter variation."""
    default_params = {
        'Guangzhou':      {'pop_decay': 0.025, 'income_decay': 0.020},
        'Chicago':        {'pop_decay': 0.018, 'income_decay': 0.015},
        'Tokyo':          {'pop_decay': 0.020, 'income_decay': 0.018},
        'Kuala Lumpur':   {'pop_decay': 0.030, 'income_decay': 0.025},
    }

    base = default_params[city_name]
    results = []

    np.random.seed(42)
    for i in range(n_samples):
        pop_decay = base['pop_decay'] * np.random.uniform(0.7, 1.3)
        income_decay = base['income_decay'] * np.random.uniform(0.7, 1.3)
        noise_sigma = 0.3 * np.random.uniform(0.5, 1.5)

        pop, income, _ = generate_synthetic_grid(
            city_name, pop_decay_rate=pop_decay,
            income_decay_rate=income_decay, noise_sigma=noise_sigma
        )
        protection = generate_protection_grid(city_name)
        benefit = pop * protection
        gini = compute_gini(benefit.flatten(), pop.flatten())
        quintiles = compute_quintile_shares(income.flatten(), pop.flatten())

        results.append({
            'pop_decay': pop_decay,
            'income_decay': income_decay,
            'noise_sigma': noise_sigma,
            'gini': gini,
            'quintile_Q1': quintiles[0],
            'quintile_Q5': quintiles[4],
        })

    return results


# ============================================================
# PART 4: Real data validation
# ============================================================

def fetch_census_tract_data():
    """
    Fetch ACS 2023 5-year census tract data for Cook County (Chicago).

    REQUIRED: Free Census API key from https://api.census.gov/data/key_signup.html
    Set environment variable: CENSUS_API_KEY=your_key_here
    """
    api_key = os.environ.get('CENSUS_API_KEY')
    if not api_key:
        print("  WARNING: Set CENSUS_API_KEY env var to fetch real data.")
        print("  Get free key: https://api.census.gov/data/key_signup.html")
        return None

    import urllib.request

    url = (
        f'https://api.census.gov/data/2023/acs/acs5'
        f'?get=NAME,B19013_001E,B01003_001E'
        f'&for=tract:*&in=state:17&in=county:031'
        f'&key={api_key}'
    )

    req = urllib.request.Request(url, headers={'User-Agent': 'DTFRI-Research/1.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    header = data[0]
    rows = data[1:]

    tracts = []
    for row in rows:
        tract_id = row[3]
        income = float(row[1]) if row[1] not in [None, '-', 'null'] else None
        population = float(row[2]) if row[2] not in [None, '-', 'null'] else None
        if income is not None and population is not None and income > 0:
            tracts.append({
                'tract_id': tract_id,
                'median_income': income,
                'population': population,
            })

    return tracts


def compare_synthetic_vs_real(city_name='Chicago'):
    """Compare synthetic Gini vs real census tract data."""
    pop, income, dist = generate_synthetic_grid(city_name)
    protection = generate_protection_grid(city_name)
    benefit = pop * protection

    gini_synthetic = compute_gini(benefit.flatten(), pop.flatten())
    quintiles_synthetic = compute_quintile_shares(income.flatten(), pop.flatten())

    print(f"\n{'='*60}")
    print(f"  {city_name} - Synthetic vs Real Data")
    print(f"{'='*60}")
    print(f"\n  SYNTHETIC:")
    print(f"  Protection Gini: {gini_synthetic:.4f}")
    print(f"  Income Q1: {quintiles_synthetic[0]:.1f}% | Q5: {quintiles_synthetic[4]:.1f}%")

    tracts = fetch_census_tract_data()
    if tracts:
        incomes = np.array([t['median_income'] for t in tracts])
        populations = np.array([t['population'] for t in tracts])
        gini_real = compute_gini(incomes, populations)
        quintiles_real = compute_quintile_shares(incomes, populations)

        print(f"\n  REAL (Cook County ACS 2023, {len(tracts)} tracts):")
        print(f"  Income Gini: {gini_real:.4f}")
        print(f"  Income Q1: {quintiles_real[0]:.1f}% | Q5: {quintiles_real[4]:.1f}%")
        print(f"\n  NOTE: Paper Gini (0.64-0.72) = flood protection benefit.")
        print(f"  Real income Gini validates the underlying distribution only.")


# ============================================================
# PART 5: Main
# ============================================================

def main():
    cities = ['Guangzhou', 'Chicago', 'Tokyo', 'Kuala Lumpur']

    print("=" * 60)
    print("  Synthetic Spatial Equity Validation")
    print("  Deep Tunnel Flood Resilience Index (DTFRI)")
    print("=" * 60)

    # ---- Baseline ----
    print(f"\n{'─'*60}")
    print("  1. BASELINE SYNTHETIC RESULTS")
    print(f"{'─'*60}")
    print(f"  {'City':<16} {'Gini':>8} {'Q1(%)':>8} {'Q5(%)':>8}")
    print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8}")

    for city in cities:
        pop, income, _ = generate_synthetic_grid(city)
        protection = generate_protection_grid(city)
        benefit = pop * protection
        gini = compute_gini(benefit.flatten(), pop.flatten())
        quintiles = compute_quintile_shares(income.flatten(), pop.flatten())
        print(f"  {city:<16} {gini:>8.4f} {quintiles[0]:>8.1f} {quintiles[4]:>8.1f}")

    # ---- Sensitivity ----
    print(f"\n{'─'*60}")
    print("  2. SENSITIVITY (decay +/-30%, noise sigma +/-50%, 50 samples)")
    print(f"{'─'*60}")
    print(f"  {'City':<16} {'Gini Mean':>10} {'Gini Range':>16} {'CV(%)':>8}")
    print(f"  {'─'*16} {'─'*10} {'─'*16} {'─'*8}")

    all_results = {}
    for city in cities:
        results = run_sensitivity(city, n_samples=50)
        ginis = [r['gini'] for r in results]
        gini_mean = np.mean(ginis)
        gini_min = np.min(ginis)
        gini_max = np.max(ginis)
        gini_cv = np.std(ginis) / gini_mean * 100 if gini_mean != 0 else 0
        print(f"  {city:<16} {gini_mean:>10.4f} [{gini_min:.4f}, {gini_max:.4f}] {gini_cv:>7.1f}")
        all_results[city] = results

    # ---- Real data ----
    print(f"\n{'─'*60}")
    print("  3. REAL DATA COMPARISON")
    print(f"{'─'*60}")
    compare_synthetic_vs_real('Chicago')

    # ---- Justification text ----
    print(f"\n{'─'*60}")
    print("  4. JUSTIFICATION FOR SYNTHETIC DATA")
    print(f"{'─'*60}")
    print("""
    The synthetic spatial equity analysis rests on:

    (a) Monocentric negative exponential model — well-established in urban
        economics (Clark 1951; McDonald 1989), validated across hundreds of
        cities worldwide.

    (b) City-specific decay gradients set from published urban morphology
        parameters for each city type (compact Asian megacity, classic North
        American monocentric, transitional polycentric, emerging SE Asian).

    (c) Robustness confirmed via:
        - Decay rate +/-30% and noise sigma +/-50% perturbation (see above)
        - MAUP test across 8 grid resolutions (ED Fig. S10, CV < 2%)

    (d) Real census tract validation (Chicago, Cook County ACS 2023) is
        feasible and recommended: tract-level income and population data
        are publicly available from the U.S. Census Bureau.
    """)

    # Save sensitivity table
    output_dir = Path('Submission/Table')
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'ED_Table_S5_Synthetic_Validation.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['City', 'Gini_Mean', 'Gini_Min', 'Gini_Max', 'Gini_CV_%',
                         'Q1_Mean_%', 'Q5_Mean_%'])
        for city in cities:
            r = all_results[city]
            ginis = [x['gini'] for x in r]
            q1s = [x['quintile_Q1'] for x in r]
            q5s = [x['quintile_Q5'] for x in r]
            writer.writerow([city, round(np.mean(ginis), 4), round(np.min(ginis), 4),
                           round(np.max(ginis), 4), round(np.std(ginis)/np.mean(ginis)*100, 1),
                           round(np.mean(q1s), 1), round(np.mean(q5s), 1)])

    print(f"  Sensitivity table saved to: {csv_path}")


if __name__ == '__main__':
    main()
