from __future__ import annotations
from dataclasses import dataclass
from math import pi, sin, radians, acos, cos, log10, sqrt, log, exp

# -----------------------------
# Physical constants & defaults
# -----------------------------
G_EARTH = 9.80665                # m/s^2
R_EARTH = 6_371_000.0            # m
P_AMBIENT = 101325.0             # Pa
C_SOUND = 330.0                  # m/s  (EIEP approx)
J_PER_MT_TNT = 4.18e15           # J in 1 megaton TNT
J_PER_KT_TNT = 4.18e12           # J in 1 kiloton TNT
D_COMPLEX_TRANSIENT_KM = 2.56
D_COMPLEX_FINAL_KM = 3.2
K_TRANSIENT_ROCK = 1.161
K_TRANSIENT_WATER = 1.365

# EIEP air-blast (1 kt surface burst) shape parameters (Eq. 54*)
PX = 75_000.0  # Pa  crossover overpressure
RX = 290.0     # m   crossover distance

# Table 4 (1 kt) characteristic distances (m) for quick scaling (Glasstone & Dolan)
TABLE4_D1 = {
    "buildings_collapse": 389.0,  # multistory wall-bearing collapse
    "homes_collapse":     502.0,  # wood frame buildings almost completely collapse
    "glass_shatter":     1160.0,
}

# Target material reference densities (kg/m^3)
TARGET_DENSITIES = {"crystalline": 2750.0, "sedimentary": 2500.0, "water": 1000.0}

# --- Atmospheric entry model constants (EIEP/ARMOR) ---
H0 = 8000.0       # m, scale height
RHO0 = 1.225      # kg/m^3, sea-level reference density
CD_ENTRY = 1.3    # effective drag coeff. (ARMOR/EIEP range)


@dataclass(frozen=True)
class Projectile:
    diameter_m: float
    speed_mps: float
    density_kgpm3: float
    angle_deg: float  # to HORIZONTAL

    @property
    def radius_m(self) -> float:
        return 0.5 * self.diameter_m

    @property
    def mass_kg(self) -> float:
        return (pi / 6.0) * self.density_kgpm3 * self.diameter_m**3

    @property
    def angle_rad(self) -> float:
        return radians(self.angle_deg)


@dataclass(frozen=True)
class Target:
    kind: str = "crystalline"
    density_kgpm3: float | None = None
    gravity_mps2: float = G_EARTH

    def rho_t(self) -> float:
        if self.density_kgpm3 is not None:
            return float(self.density_kgpm3)
        k = self.kind.lower()
        if k not in TARGET_DENSITIES:
            raise ValueError(f"Unknown target kind '{self.kind}'.")
        return TARGET_DENSITIES[k]

    def transient_prefactor(self) -> float:
        return K_TRANSIENT_WATER if self.kind.lower() == "water" else K_TRANSIENT_ROCK


class ImpactModel:
    """
    Crater + thermal + seismic + air-blast + recurrence + atmospheric entry + tsunami.
    Water impacts: Water Layer deceleration → use E_surface vs E_seafloor appropriately.
    """

    def __init__(self, projectile: Projectile, target: Target):
        self.p = projectile
        self.t = target

    # ---------- Energetics ----------
    def kinetic_energy_J(self) -> float:
        return 0.5 * self.p.mass_kg * self.p.speed_mps**2

    def energy_mt_tnt(self) -> float:
        return self.kinetic_energy_J() / J_PER_MT_TNT

    def energy_kt_tnt(self) -> float:
        return self.kinetic_energy_J() / J_PER_KT_TNT

    # ---------- ATMOSPHERIC ENTRY (EIEP/ARMOR) ----------
    def _eiep_strength_pa(self) -> float:
        """EIEP empirical strength [Pa] from bulk density: log10(Y)=2.107+0.0624*sqrt(rho_i)"""
        rho_i = max(self.p.density_kgpm3, 1.0)
        return 10.0 ** (2.107 + 0.0624 * sqrt(rho_i))

    def _rho_at(self, z_m: float) -> float:
        """Exponential atmosphere: rho(z) = RHO0 * exp(-z/H0)."""
        return RHO0 * exp(-max(z_m, 0.0)/H0)

    def breakup_altitude_m(self) -> tuple[str, float | None]:
        """
        Breakup state + z* (m). If>=1 → 'intact', else 'fragmenting' with z*.
        """
        Y = self._eiep_strength_pa()
        rho_i = self.p.density_kgpm3
        L0 = self.p.diameter_m
        v0 = self.p.speed_mps
        sT = max(1e-6, sin(self.p.angle_rad))

        If = (4.07 * CD_ENTRY * H0 * Y) / (rho_i * L0 * v0*v0 * sT)
        if If >= 1.0:
            return "intact", None

        root = sqrt(max(0.0, 1.0 - If))
        z_star = -H0 * (log(Y/(RHO0*v0*v0)) + 1.308 - 0.314*If - 1.303*root)
        return "fragmenting", z_star

    def airburst_altitude_m(self, fp: float = 7.0) -> float | None:
        """Airburst altitude z_b (m) from ARMOR/EIEP closed form; None if no fragmentation."""
        mode, z_star = self.breakup_altitude_m()
        if mode != "fragmenting" or z_star is None:
            return None

        rho_zs = self._rho_at(z_star)
        L0 = self.p.diameter_m
        sT = max(1e-6, sin(self.p.angle_rad))
        rho_i = self.p.density_kgpm3

        # Dispersion length at breakup (ARMOR)
        Ldisp = L0 * sT * sqrt(rho_i / (CD_ENTRY * rho_zs))
        alpha = sqrt(max(0.0, fp*fp - 1.0))
        zb = z_star - 2.0*H0 * log(1.0 + (Ldisp/(2.0*H0)) * alpha)
        return zb

    def entry_outcome(self) -> dict:
        """
        'surface_impact' or 'airburst' + z*, z_b, pancake factor.
        """
        mode, z_star = self.breakup_altitude_m()
        if mode == "intact":
            return {"mode": "surface_impact", "breakup_altitude_m": None, "airburst_altitude_m": None, "pancake_factor": 7.0}
        zb = self.airburst_altitude_m(fp=7.0)
        if zb is not None and zb > 0.0:
            return {"mode": "airburst", "breakup_altitude_m": z_star, "airburst_altitude_m": zb, "pancake_factor": 7.0}
        else:
            return {"mode": "surface_impact", "breakup_altitude_m": z_star, "airburst_altitude_m": zb, "pancake_factor": 7.0}

    # ---------- Water Layer deceleration (EIEP Eq. ~65*) ----------
    def water_layer_velocities(self, water_depth_m: float = 3682.0) -> dict:
        """
        Returns v_surface (≈input), v_seafloor, E_surface, E_seafloor for water targets.
        """
        if self.t.kind.lower() != "water":
            return {}
        rho_w = 1000.0
        Cdw  = 0.877  # EIEP choice
        dw   = max(water_depth_m, 0.0)
        L    = self.p.diameter_m
        Ui   = self.p.density_kgpm3
        sT   = max(1e-6, sin(self.p.angle_rad))

        v_surface = self.p.speed_mps
        factor = - (3.0 * rho_w * Cdw * dw) / (2.0 * Ui * L * sT)
        v_seafloor = v_surface * exp(factor)

        m = self.p.mass_kg
        E_surface  = 0.5 * m * v_surface**2
        E_seafloor = 0.5 * m * v_seafloor**2

        return {"v_surface": v_surface, "v_seafloor": v_seafloor,
                "E_surface": E_surface, "E_seafloor": E_seafloor}

    # ---------- Crater scaling (generic land/water surface) ----------
    def transient_diameter_m(self, v_override_mps: float | None = None,
                             rho_t_override: float | None = None,
                             K_override: float | None = None) -> float:
        """
        Allow overrides so water-floor crater can use v_seafloor + rock target.
        """
        rho_i = self.p.density_kgpm3
        rho_t = self.t.rho_t() if rho_t_override is None else rho_t_override
        L0 = self.p.diameter_m
        v = self.p.speed_mps if (v_override_mps is None) else v_override_mps
        g = self.t.gravity_mps2
        gamma = self.p.angle_rad
        K = (self.t.transient_prefactor() if K_override is None else K_override)
        return K * (rho_i / rho_t) ** (1.0 / 3.0) * (L0 ** 0.78) * (v ** 0.44) * (g ** -0.22) * (sin(gamma) ** (1.0 / 3.0))

    def _final_from_transient(self, Dtc_m: float) -> tuple[float, float]:
        """Final diameter & depth from transient diameter."""
        Dtc_km = Dtc_m / 1000.0
        if Dtc_km < D_COMPLEX_TRANSIENT_KM:
            Dfr_m = 1.25 * Dtc_m
        else:
            Dfr_km = 1.17 * (Dtc_km ** 1.13) * (D_COMPLEX_FINAL_KM ** -0.13)
            Dfr_m = Dfr_km * 1000.0
        dfr_km = 0.294 * ((Dfr_m/1000.0) ** 0.301)
        return Dfr_m, dfr_km*1000.0

    def water_craters(self, water_depth_m: float = 3682.0) -> dict:
        """
        Two craters for water impact:
          - in-water transient (water properties, v_surface)
          - seafloor transient/final (rock properties, v_seafloor)
        """
        w = self.water_layer_velocities(water_depth_m)
        if not w:
            return {}

        # water column crater (in-water)
        Dtc_water = self.transient_diameter_m(
            v_override_mps=w["v_surface"],
            rho_t_override=1000.0,
            K_override=K_TRANSIENT_WATER,
        )

        # seafloor crater (rock target) using v_seafloor
        Dtc_floor = self.transient_diameter_m(
            v_override_mps=w["v_seafloor"],
            rho_t_override=2700.0,
            K_override=K_TRANSIENT_ROCK,
        )
        Dfr_floor, dfr_floor = self._final_from_transient(Dtc_floor)

        return {"Dtc_water_m": Dtc_water,
                "Dtc_seafloor_m": Dtc_floor,
                "Dfr_seafloor_m": Dfr_floor,
                "dfr_seafloor_m": dfr_floor,
                "energies": w}

    def final_diameter_m(self) -> float:
        # Gate by entry outcome – no crater for airburst
        if self.entry_outcome()["mode"] == "airburst":
            return 0.0
        # Land or water-surface generic crater
        Dtc_m = self.transient_diameter_m()
        Dfr_m, _ = self._final_from_transient(Dtc_m)
        return Dfr_m

    def final_depth_m(self) -> float:
        if self.entry_outcome()["mode"] == "airburst":
            return 0.0
        Dfr_m = self.final_diameter_m()
        dfr_km = 0.294 * ((Dfr_m/1000.0) ** 0.301)
        return dfr_km * 1000.0

    # ---------- Recurrence ----------
    def global_recurrence_years(self) -> float:
        E_mt = self.energy_mt_tnt()
        return float("inf") if E_mt <= 0.0 else (109 * (E_mt ** 0.78))

    # ---------- Thermal radiation (allow energy override) ----------
    def fireball_radius_m(self, energy_J: float | None = None) -> float:
        E = self.kinetic_energy_J() if energy_J is None else energy_J
        return 0.002 * (E ** (1.0/3.0))

    def _horizon_fraction(self, r_m: float, Rf_m: float) -> float:
        phi = r_m / R_EARTH
        h = (1.0 - cos(phi)) * R_EARTH
        if h >= Rf_m:
            return 0.0
        Gamma = acos(h / Rf_m)
        return (2.0/pi) * (Gamma - (h/Rf_m)*sin(Gamma))

    def thermal_exposure_Jpm2(self, r_m: float, luminous_eff=3e-3, energy_J: float | None = None,
                              burst_alt_m: float | None = None) -> float:
        """
        Φ(r)=f*K*E/(2π R^2), ahol R=slant range: R= sqrt(r^2 + zb^2) ha airburst, különben r.
        """
        E = self.kinetic_energy_J() if energy_J is None else energy_J
        Rf = self.fireball_radius_m(energy_J=E)
        # slant range az airburst miatt:
        if burst_alt_m is not None and burst_alt_m > 0.0:
            R = (r_m**2 + burst_alt_m**2) ** 0.5
        else:
            R = r_m
        f = self._horizon_fraction(R, Rf)
        return f * luminous_eff * E / (2.0 * pi * R**2)

    def _scaled_threshold_Jpm2(self, mj_per_m2_at_1mt: float, energy_Mt: float | None = None) -> float:
        E_Mt = self.energy_mt_tnt() if energy_Mt is None else energy_Mt
        return (mj_per_m2_at_1mt * 1e6) * (E_Mt ** (1.0/6.0))

    def distances_for_thermal_rings(self, luminous_eff=3e-3,
                                    energy_J: float | None = None,
                                    burst_alt_m: float | None = None) -> dict:
        tbl = {
            "clothes_ignite": 1.00,
            "plywood_ignite": 0.67,
            "grass_ignite": 0.38,
            "newspaper_ignite": 0.33,
            "trees_ignite": 0.25,
            "third_degree_burn": 0.42,
            "second_degree_burn": 0.25,
            "first_degree_burn": 0.13,
        }
        E = self.kinetic_energy_J() if energy_J is None else energy_J
        E_Mt = E / J_PER_MT_TNT
        thr = {k: self._scaled_threshold_Jpm2(v, energy_Mt=E_Mt) for k, v in tbl.items()}

        def solve_for_r(target):
            lo, hi = 1e3, 1_000_000.0
            for _ in range(70):
                mid = 0.5*(lo+hi)
                if self.thermal_exposure_Jpm2(mid, luminous_eff, energy_J=E, burst_alt_m=burst_alt_m) > target:
                    lo = mid
                else:
                    hi = mid
            return hi/1000.0
        return {k: solve_for_r(v) for k, v in thr.items()}

    # ---------- AIR BLAST (allow energy override; surface-burst shape) ----------
    def overpressure_pa_surface(self, r_m: float, energy_kt: float | None = None) -> float:
        Ekt = self.energy_kt_tnt() if energy_kt is None else max(energy_kt, 1e-9)
        r1 = r_m / (Ekt ** (1.0/3.0))
        return (PX * (RX / (4.0 * r1))) / (1.0 + 3.0 * (r1 / RX) ** 1.3)

    def wind_speed_mps(self, r_m: float, energy_kt: float | None = None) -> float:
        p = self.overpressure_pa_surface(r_m, energy_kt=energy_kt)
        return (5.0 * p / (7.0 * P_AMBIENT)) * C_SOUND * (1.0 + 6.0 * p / (7.0 * P_AMBIENT)) ** 0.5

    def _solve_radius_for_overpressure(self, p_target: float, energy_kt: float | None = None) -> float | None:
        lo, hi = 1e3, 2_000_000.0
        for _ in range(80):
            mid = 0.5*(lo+hi)
            if self.overpressure_pa_surface(mid, energy_kt=energy_kt) > p_target:
                lo = mid
            else:
                hi = mid
        r = hi
        return r/1000.0 if self.overpressure_pa_surface(r, energy_kt=energy_kt) <= p_target else None

    def _solve_radius_for_wind(self, v_target_mps: float, energy_kt: float | None = None) -> float | None:
        lo, hi = 1e3, 2_000_000.0
        for _ in range(80):
            mid = 0.5*(lo+hi)
            if self.wind_speed_mps(mid, energy_kt=energy_kt) > v_target_mps:
                lo = mid
            else:
                hi = mid
        r = hi
        return r/1000.0 if self.wind_speed_mps(r, energy_kt=energy_kt) <= v_target_mps else None

    def table4_scaled_radii_km(self, energy_kt: float | None = None) -> dict:
        Ekt = self.energy_kt_tnt() if energy_kt is None else max(energy_kt, 1e-9)
        s = Ekt ** (1.0/3.0)
        return {k: (d1 * s) / 1000.0 for k, d1 in TABLE4_D1.items()}

    def health_rings_km(self, energy_kt: float | None = None) -> dict:
        return {
            "severe_lung_damage": self._solve_radius_for_overpressure(200_000.0, energy_kt=energy_kt),
            "eardrum_rupture":    self._solve_radius_for_overpressure(35_000.0,  energy_kt=energy_kt),
        }

    def fatal_radius_km(self, energy_kt: float | None = None) -> float | None:
        b = -2.424e-5
        c = -4.404e5
        def Vp(p):  # Pa → fatality prob.
            return 1.0 / (1.0 + exp(b * (p + c)))
        p_hi, p_lo = 5e6, 1e4
        for _ in range(80):
            pm = 0.5*(p_hi+p_lo)
            if Vp(pm) >= 0.99:
                p_lo = pm
            else:
                p_hi = pm
        return self._solve_radius_for_overpressure(p_lo, energy_kt=energy_kt)

    def wind_rings_km(self, energy_kt: float | None = None) -> dict:
        """Convenience wrapper: typical wind-speed thresholds radii (km)."""
        def kph_to_mps(kph): return kph / 3.6
        return {
            "wind_1500_kph": self._solve_radius_for_wind(kph_to_mps(1500.0), energy_kt=energy_kt),
            "wind_800_kph":  self._solve_radius_for_wind(kph_to_mps(800.0),  energy_kt=energy_kt),
            "EF5_320_kph":   self._solve_radius_for_wind(kph_to_mps(320.0),  energy_kt=energy_kt),
            "EF3_220_kph":   self._solve_radius_for_wind(kph_to_mps(220.0),  energy_kt=energy_kt),
            "trees_down":    self._solve_radius_for_wind(58.0,               energy_kt=energy_kt),
        }

    # ---------- Seismic ----------
    def seismic_magnitude(self, seismic_efficiency: float = 1e-4, energy_J: float | None = None) -> float:
        E = self.kinetic_energy_J() if energy_J is None else energy_J
        return 0.67 * log10(seismic_efficiency * E) - 5.87

    def effective_magnitude_at_distance_km(self, r_km: float,
                                           seismic_efficiency: float = 1e-4,
                                           energy_J: float | None = None) -> float:
        M = self.seismic_magnitude(seismic_efficiency, energy_J=energy_J)
        if r_km < 60.0:
            return M - 0.0238 * r_km
        elif r_km < 700.0:
            return M - 0.0048 * r_km - 1.1644
        else:
            Delta = (r_km * 1000.0) / R_EARTH
            return M - 1.66 * log10(Delta) - 6.399

    def distance_for_effective_magnitude(self, Meff_target: float,
                                         seismic_efficiency: float = 1e-4,
                                         energy_J: float | None = None) -> float | None:
        M = self.seismic_magnitude(seismic_efficiency, energy_J=energy_J)
        r1 = (M - Meff_target) / 0.0238
        if 0.0 <= r1 < 60.0: return r1
        r2 = (M - 1.1644 - Meff_target) / 0.0048
        if 60.0 <= r2 < 700.0: return r2
        expo = (M - 6.399 - Meff_target) / 1.66
        r3_km = (10.0 ** expo) * (R_EARTH / 1000.0)
        return r3_km if r3_km >= 700.0 else None

    def seismic_rings(self, mags=(7.0, 6.0, 5.0, 4.0, 3.5, 3.0),
                      seismic_efficiency: float = 1e-4,
                      energy_J: float | None = None) -> dict:
        out = {}
        for Mt in mags:
            r = self.distance_for_effective_magnitude(Mt, seismic_efficiency, energy_J=energy_J)
            out[f"M{Mt:g}"] = None if r is None else {"radius_km": r, "arrival_s": r/5.0}
        return out

    def seismic_summary(self, seismic_efficiency: float = 1e-4, felt_Meff: float = 3.5,
                        energy_J: float | None = None) -> dict:
        M = self.seismic_magnitude(seismic_efficiency, energy_J=energy_J)
        def ring(Mt):
            r = self.distance_for_effective_magnitude(Mt, seismic_efficiency, energy_J=energy_J)
            if r is None: return None, None
            Ts = r / 5.0
            return r, Ts
        r6, t6 = ring(6.0)
        r5, t5 = ring(5.0)
        rfelt, tfelt = ring(felt_Meff)
        rings = self.seismic_rings(seismic_efficiency=seismic_efficiency, energy_J=energy_J)
        return {
            "source_magnitude_M": M,
            "radius_km_M6": r6, "arrival_s_M6": t6,
            "radius_km_M5": r5, "arrival_s_M5": t5,
            "felt_radius_km": rfelt, "felt_threshold_Meff": felt_Meff, "arrival_s_felt": tfelt,
            "rings": rings,
            "seismic_efficiency": seismic_efficiency,
        }

    # ---------- TSUNAMI (for water targets) ----------
    def tsunami_summary(self, water_depth_m: float = 3682.0,
                        coast_distance_km: float | None = 100.0,
                        slope_ratio: float = 0.005) -> dict:
        """
        Near-field rim/collapse + optional far-field deep-water amplitude and run-up.
        If coast_distance_km is None → omit far-field estimates.
        """
        # Use in-water transient diameter for wave scaling near source
        Dtc = self.transient_diameter_m(
            v_override_mps=self.p.speed_mps, rho_t_override=1000.0, K_override=K_TRANSIENT_WATER
        )
        L0 = self.p.diameter_m
        H = max(water_depth_m, 1.0)

        # Near-field (source-proximal) amplitudes:
        # Rim wave at ~0.75*Dtc and collapse wave at ~2.5*Dtc (deep water condition for collapse)
        Amax_rim = min(Dtc/14.1, H)         # EIEP cap
        near_rim_R = 0.75 * Dtc
        near_rim_amp = Amax_rim              # at its formation distance

        near_collapse_amp = None
        near_collapse_R = 2.5 * Dtc
        if H > 2.0 * L0:                      # deep water criterion for collapse wave
            dtc = Dtc / 2.828
            Amax_cw = 0.06 * min(dtc, H)
            near_collapse_amp = Amax_cw

        out = {
            "inputs": {"water_depth_m": H, "shelf_slope_ratio": slope_ratio, "coast_distance_km": coast_distance_km},
            "nearfield_rim_amp_m": near_rim_amp,
            "nearfield_rim_radius_m": near_rim_R,
            "nearfield_collapse_amp_m": near_collapse_amp,
            "nearfield_collapse_radius_m": near_collapse_R if near_collapse_amp is not None else None,
        }

        # Far-field (coast) only if distance known
        if coast_distance_km is not None:
            D = max(coast_distance_km*1000.0, 1.0)
            # Deep-water amplitude at D (∝ 1/r)
            A0 = min(0.14 * Dtc, H)
            A_deep = A0 * (Dtc / (2.0 * D))

            # Run-up via Irribaren (with xi floor to represent shelf protection)
            w = 2.0 * Dtc
            xi = slope_ratio * (2.0 * A_deep / w) ** (-0.5)
            xi = max(xi, 0.11)
            runup = 2.0 * A_deep * xi

            # Attenuated rim & collapse amplitudes at coast distance
            rim_amp_coast = Amax_rim * ( (0.75*Dtc) / max(D, 0.75*Dtc) )
            collapse_amp_coast = None
            if near_collapse_amp is not None:
                q = 1.5
                collapse_amp_coast = near_collapse_amp * ((2.5*Dtc) / max(D, 2.5*Dtc)) ** q

            out.update({
                "deep_amplitude_m": A_deep,
                "runup_m": runup,
                "rim_wave_amp_at_coast_m": rim_amp_coast,
                "collapse_wave_amp_at_coast_m": collapse_amp_coast,
            })
        return out

    # --- EJECTA (EIEP) -------------------------------------------------
    def ejecta_rim_height_m(self) -> float:
        """h_tr = D_tc / 14.1  (Eq. 48* in EIEP; D_tc in m)"""
        return self.transient_diameter_m() / 14.1

    def ejecta_thickness_m_at_rkm(self, r_km: float) -> float | None:
        """
        t_e(r) = D_tc^4 / (112 r^3)  (Eq. 47*), valid only outside the final rim.
        r_km: distance from impact (km).
        """
        if r_km <= 0.0:
            return None
        Dtc = self.transient_diameter_m()        # m
        Dfr = self.final_diameter_m()            # m (0 if airburst)
        rim_radius_km = max(Dfr, 0.0) / 2000.0   # D_fr/2 [km]
        if r_km < max(rim_radius_km, 1e-6):
            return None
        r_m = r_km * 1000.0
        te = (Dtc**4) / (112.0 * (r_m**3))
        # For small energies limit beyond fireball (EIEP limit E<200 Mt)
        if self.energy_mt_tnt() < 200.0:
            rmax_km = self.fireball_radius_m()/1000.0
            if r_km > max(rmax_km, rim_radius_km):
                return 0.0
        return te

    def ejecta_arrival_time_s(self, r_km: float) -> float | None:
        from math import tan, isfinite
        if r_km <= 0.0:
            return None
        # epicentral angle
        phi = (r_km * 1000.0) / R_EARTH
        g = self.t.gravity_mps2

        # Simplified 45° launch approximation
        tp = tan(0.5 * phi)
        ve2 = 2.0 * g * R_EARTH * (tp / (1.0 + tp))
        if ve2 <= 0.0:
            return None
        ve = sqrt(ve2)
        # validity
        if ve2 / (g * R_EARTH) >= 1.0 or r_km > 10000.0:
            return None

        # average flight time (scaled)
        e = 1.0 - 0.5 * ve2 / (g * R_EARTH)
        a = ve2 / (2.0 * g * (1.0 - e*e))
        if not isfinite(e) or not isfinite(a) or a <= 0.0:
            return None
        Te = (phi) * (a ** 1.5) / sqrt(g * R_EARTH)
        return Te if isfinite(Te) and Te > 0.0 else None

    def ejecta_summary(self, sample_radii_km: list[float] | None = None) -> dict:
        if sample_radii_km is None:
            Dfr = self.final_diameter_m()
            if Dfr > 0.0:
                base = max(Dfr/2000.0, 1.0)  # km
                sample_radii_km = [base, 2*base, 5*base, 10*base, 50*base]
            else:
                sample_radii_km = [5.0, 10.0, 25.0, 50.0, 100.0]

        rmax_km = None
        if self.energy_mt_tnt() < 200.0:
            rmax_km = self.fireball_radius_m()/1000.0

        out = {
            "rim_height_m": self.ejecta_rim_height_m(),
            "fireball_limit_km": rmax_km,
            "samples": []
        }
        for rk in sample_radii_km:
            te = self.ejecta_thickness_m_at_rkm(rk)
            Te = self.ejecta_arrival_time_s(rk)
            out["samples"].append({"r_km": rk, "thickness_m": te, "arrival_s": Te})
        return out

    # ---------- Convenience summary ----------
    def summary(self, luminous_eff=3e-3,
                tsunami_depth_m: float = 3682.0, tsunami_distance_km: float | None = 100.0,
                tsunami_slope_ratio: float = 0.005) -> dict:
        entry = self.entry_outcome()

        # Energies: default full kinetic
        E_J_full = self.kinetic_energy_J()
        E_Mt_full = E_J_full / J_PER_MT_TNT
        E_kt_full = E_J_full / J_PER_KT_TNT

        # If water: compute E_surface / E_seafloor for routing
        E_surface = E_J_full
        E_seafloor = E_J_full
        water_info = None
        if self.t.kind.lower() == "water":
            water_info = self.water_layer_velocities(tsunami_depth_m)
            if water_info:
                E_surface = water_info["E_surface"]
                E_seafloor = water_info["E_seafloor"]

        # CRATER
        if self.t.kind.lower() == "water":
            wcr = self.water_craters(tsunami_depth_m)
            if entry["mode"] == "airburst":
                crater_block = {"transient_diameter_m": 0.0, "final_diameter_m": 0.0, "final_depth_m": 0.0,
                                "regime": "airburst (no crater)",
                                "water_craters": wcr}
            else:
                crater_block = {"transient_diameter_m": wcr.get("Dtc_seafloor_m", 0.0),
                                "final_diameter_m": wcr.get("Dfr_seafloor_m", 0.0),
                                "final_depth_m": wcr.get("dfr_seafloor_m", 0.0),
                                "regime": "seafloor (rock) crater",
                                "water_craters": wcr}
        else:
            if entry["mode"] == "airburst":
                crater_block = {"transient_diameter_m": 0.0, "final_diameter_m": 0.0, "final_depth_m": 0.0,
                                "regime": "airburst (no crater)"}
            else:
                Dtc = self.transient_diameter_m()
                Dfr, dfr = self._final_from_transient(Dtc)
                crater_block = {"transient_diameter_m": Dtc, "final_diameter_m": Dfr, "final_depth_m": dfr,
                                "regime": ("simple" if (Dtc/1000.0) < D_COMPLEX_TRANSIENT_KM else "complex")}

        # THERMAL/BLAST energy routing:
        zb = entry.get("airburst_altitude_m", None) if entry["mode"] == "airburst" else None

        thermal_rings = self.distances_for_thermal_rings(
            luminous_eff=luminous_eff,
            energy_J=(E_surface if self.t.kind.lower()=="water" else E_J_full),
            burst_alt_m=zb
        )

        energy_kt_for_blast = ((E_surface/J_PER_KT_TNT) if self.t.kind.lower()=="water" else E_kt_full)
        air_health = self.health_rings_km(energy_kt=energy_kt_for_blast)
        air_tbl4   = self.table4_scaled_radii_km(energy_kt=energy_kt_for_blast)

        # re-run wind with energy override:
        wind = {
            "wind_1500_kph": self._solve_radius_for_wind(1500.0/3.6, energy_kt=energy_kt_for_blast),
            "wind_800_kph":  self._solve_radius_for_wind(800.0/3.6,  energy_kt=energy_kt_for_blast),
            "EF5_320_kph":   self._solve_radius_for_wind(320.0/3.6,  energy_kt=energy_kt_for_blast),
            "EF3_220_kph":   self._solve_radius_for_wind(220.0/3.6,  energy_kt=energy_kt_for_blast),
            "trees_down":    self._solve_radius_for_wind(58.0,       energy_kt=energy_kt_for_blast),
        }

        # SEISMIC with E_seafloor if water impact
        seis_energy = E_seafloor if self.t.kind.lower()=="water" else E_J_full
        seis = self.seismic_summary(energy_J=seis_energy)

        # TSUNAMI if water
        out = {}
        if self.t.kind.lower() == "water":
            out["tsunami"] = self.tsunami_summary(
                water_depth_m=tsunami_depth_m,
                coast_distance_km=tsunami_distance_km,
                slope_ratio=tsunami_slope_ratio
            )

        ejecta = self.ejecta_summary()
        out["ejecta"] = ejecta

        return {
            "entry": entry,
            "projectile": {"diameter_m": self.p.diameter_m, "speed_mps": self.p.speed_mps,
                           "density_kgpm3": self.p.density_kgpm3, "angle_deg": self.p.angle_deg,
                           "mass_kg": self.p.mass_kg},
            "target": {"kind": self.t.kind, "density_kgpm3": self.t.rho_t(), "gravity_mps2": self.t.gravity_mps2},
            "energy": {"kinetic_J": E_J_full, "tnt_megatons": E_Mt_full, "tnt_kilotons": E_kt_full*1e3},
            "crater": crater_block,
            "thermal": {"fireball_diameter_km": 2.0 * self.fireball_radius_m(energy_J=(E_surface if self.t.kind.lower()=="water" else E_J_full)) / 1000.0,
                        "ring_distances_km": thermal_rings, "luminous_efficiency": luminous_eff},
            "air_blast": {
                "health_rings_km": air_health,
                "table4_rings_km": air_tbl4,
                "fatal_radius_km": self.fatal_radius_km(energy_kt=energy_kt_for_blast),
                "wind_rings_km": wind,
                "spl_at_1km_db": 20.0*log10(max(self.overpressure_pa_surface(1000.0, energy_kt=energy_kt_for_blast),1.0)/20e-6),
                "energy_kt_used": energy_kt_for_blast,
            },
            "seismic": seis,
            "frequency": {"global_recurrence_years": self.global_recurrence_years()},
            **out
        }


def mmi_band_from_meff(meff: float) -> str:
    if meff < 1.0:   return "-"
    if meff < 2.0:   return "I"
    if meff < 3.0:   return "I–II"
    if meff < 4.0:   return "III–IV"
    if meff < 5.0:   return "IV–V"
    if meff < 6.0:   return "VI–VII"
    if meff < 7.0:   return "VII–VIII"
    if meff < 8.0:   return "IX–X"
    if meff < 9.0:   return "X–XI"
    return "XII"
