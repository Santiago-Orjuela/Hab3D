"""
Módulo para calcular perfiles geotérmicos de planetas tipo Tierra

Este módulo implementa un modelo de conducción térmica 1D para la litosfera/corteza
basado en Hasterok & Chapman (2011) y otros trabajos sobre geofísica térmica.

Características principales:
- Ecuación de conducción térmica en estado estacionario
- Producción radiactiva con decaimiento exponencial (Hasterok model)
- Conductividad térmica con promedio Voigt-Reuss-Hill (VRH)
- Acoplamiento presión-temperatura-densidad usando BurnMan
- Gravedad local variable con la profundidad
- **NUEVO**: Detección automática de límites de temperatura de BurnMan
- **NUEVO**: Modelos de evolución temporal del gradiente geotérmico

Detección de Límites:
---------------------
La función calculate_geotherm() ahora incluye detección automática de límites
de temperatura (T_max_safe, default 2150 K). Cuando la temperatura calculada
excede este límite, el cálculo se detiene y retorna un perfil parcial con
solo las capas calculadas exitosamente. Esto previene fallos de BurnMan en
regímenes de alta temperatura (típicamente para épocas antiguas con alto
flujo de calor).

Evolución Temporal:
------------------
Nuevas funciones para modelar cómo varía el gradiente geotérmico a lo largo
del tiempo geológico:
- q_s_turcotte(): Modelo de flujo de calor temporal (Turcotte & Schubert 2014)
- A_surface_temporal(): Producción radiactiva temporal
- calculate_geotherm_evolution(): Calcula perfiles T(z) para múltiples épocas

Funciones Principales:
---------------------
- calculate_geotherm(): Calcula perfil T(z), P(z), ρ(z) para un tiempo dado
- calculate_geotherm_evolution(): Evolución temporal del gradiente geotérmico
- prepare_rocks_dict(): Crea Composites de BurnMan desde composiciones modales
- get_mineral_objects(): Obtiene objetos minerales de BurnMan

Ejemplo de Uso Rápido:
----------------------
>>> from geotherm_calculator import *
>>> mineral_objects = get_mineral_objects()
>>> rocks = prepare_rocks_dict(mineral_objects=mineral_objects)
>>> 
>>> # Perfil para la Tierra actual
>>> df = calculate_geotherm(rocks=rocks, q_s=65e-3, z_max=100e3,
...                         dz=100, R_planet=Re, M_total=Me)
>>> 
>>> # Evolución temporal (0-1 Ga)
>>> results = calculate_geotherm_evolution(rocks=rocks,
...                                       composition=COMPOSITION_DEFAULT,
...                                       n_times=10, t_range=(0, 1.0))

Autor: Santiago
Fecha: Octubre 2025
Última actualización: Octubre 31, 2025
Basado en: Hasterok & Chapman (2011), Turcotte & Schubert (2014)
"""

import math
import numpy as np
import pandas as pd
from astropy import constants
from burnman import minerals, Composite

# =============================================================================
# CONSTANTES FÍSICAS
# =============================================================================
G = constants.G.value  # Constante gravitacional
Me = constants.M_earth.value  # Masa de la Tierra (kg)
Re = constants.R_earth.value  # Radio de la Tierra (m)
km = 1000.0  # metros por kilómetro


# =============================================================================
# COMPOSICIONES MODALES DE CAPAS (% volumétrico)
# =============================================================================

# Composiciones basadas en estudios petrológicos de corteza continental
COMPOSITION_DEFAULT = {
    "upper": {
        "Quartz": 20, "Orthoclase": 15, "Albite": 32, "Anorthite": 8,
        "Phlogopite": 10, "Hornblende": 13, "Diopside": 1, "Hedenbergite": 1,
        "Enstatite": 2, "Ferrosillite": 3, "Forsterite": 64.04, "Fayalite": 4.96,
        "Pyrope": 3.19, "Almandine": 9
    },
    "middle": {
        "Quartz": 15, "Orthoclase": 5, "Albite": 35, "Anorthite": 20,
        "Hornblende": 20, "Diopside": 1.96, "Hedenbergite": 0.14,
        "Enstatite": 1, "Ferrosillite": 1, "Forsterite": 63.65, "Fayalite": 3.35,
        "Pyrope": 5.58, "Almandine": 0.81
    },
    "lower": {
        "Quartz": 2, "Orthoclase": 10, "Albite": 10, "Anorthite": 18,
        "Hornblende": 47, "Diopside": 5.47, "Hedenbergite": 0.53,
        "Enstatite": 23.22, "Ferrosillite": 1.78, "Forsterite": 54.20,
        "Fayalite": 5.80, "Pyrope": 9.57, "Almandine": 1.42
    },
    "mantle": {
        "Forsterite": 64.04, "Fayalite": 4.96, "Enstatite": 15.47,
        "Ferrosillite": 1.53, "Diopside": 9.97, "Pyrope": 0, "Almandine": 2.43
    }
}


# =============================================================================
# PARÁMETROS DE CONDUCTIVIDAD TÉRMICA POR MINERAL
# =============================================================================

# Basados en Hofmeister (1999), Stackhouse (2015), Hasterok & Chapman (2011)
MINERAL_PARAMS = {
    "Quartz": {"lambda0": 6.5, "n": 0.5, "KT": 60.0, "KTp": 4.0,
               "lambdaRmax": 0.5, "TR": 1400.0, "omega": 300.0},
    "Orthoclase": {"lambda0": 2.5, "n": 0.5, "KT": 60.0, "KTp": 4.0,
                   "lambdaRmax": 0.2, "TR": 1400.0, "omega": 300.0},
    "Albite": {"lambda0": 3.0, "n": 0.5, "KT": 60.0, "KTp": 4.0,
               "lambdaRmax": 0.3, "TR": 1400.0, "omega": 300.0},
    "Anorthite": {"lambda0": 3.5, "n": 0.5, "KT": 70.0, "KTp": 4.0,
                  "lambdaRmax": 0.4, "TR": 1400.0, "omega": 300.0},
    "Phlogopite": {"lambda0": 1.7, "n": 0.4, "KT": 40.0, "KTp": 4.0,
                   "lambdaRmax": 0.1, "TR": 1300.0, "omega": 300.0},
    "Hornblende": {"lambda0": 2.2, "n": 0.5, "KT": 60.0, "KTp": 4.0,
                   "lambdaRmax": 0.5, "TR": 1350.0, "omega": 300.0},
    "Diopside": {"lambda0": 3.8, "n": 0.5, "KT": 105.0, "KTp": 4.3,
                 "lambdaRmax": 1.0, "TR": 1400.0, "omega": 300.0},
    "Hedenbergite": {"lambda0": 3.0, "n": 0.5, "KT": 100.0, "KTp": 4.0,
                     "lambdaRmax": 0.8, "TR": 1350.0, "omega": 300.0},
    "Enstatite": {"lambda0": 4.2, "n": 0.5, "KT": 107.0, "KTp": 4.5,
                  "lambdaRmax": 1.5, "TR": 1300.0, "omega": 300.0},
    "Ferrosillite": {"lambda0": 3.5, "n": 0.5, "KT": 100.0, "KTp": 4.0,
                     "lambdaRmax": 1.0, "TR": 1300.0, "omega": 300.0},
    "Forsterite": {"lambda0": 5.5, "n": 0.6, "KT": 128.0, "KTp": 4.2,
                   "lambdaRmax": 3.0, "TR": 1200.0, "omega": 300.0},
    "Fayalite": {"lambda0": 4.5, "n": 0.5, "KT": 130.0, "KTp": 4.0,
                 "lambdaRmax": 1.5, "TR": 1200.0, "omega": 300.0},
    "Pyrope": {"lambda0": 4.0, "n": 0.6, "KT": 171.0, "KTp": 4.3,
               "lambdaRmax": 2.0, "TR": 1400.0, "omega": 300.0},
    "Almandine": {"lambda0": 3.8, "n": 0.5, "KT": 175.0, "KTp": 4.5,
                  "lambdaRmax": 1.5, "TR": 1400.0, "omega": 300.0}
}


# =============================================================================
# OBJETOS MINERALES DE BURNMAN
# =============================================================================

def get_mineral_objects():
    """
    Crea diccionario con objetos de minerales de BurnMan.
    
    Returns
    -------
    dict
        Diccionario {nombre_mineral: objeto_BurnMan}
    """
    return {
        "Quartz": minerals.SLB_2011.qtz(),
        "Albite": minerals.SLB_2011.albite(),
        "Anorthite": minerals.SLB_2011.anorthite(),
        "Diopside": minerals.SLB_2011.diopside(),
        "Hedenbergite": minerals.SLB_2011.hedenbergite(),
        "Enstatite": minerals.SLB_2011.enstatite(),
        "Ferrosillite": minerals.SLB_2011.ferrosilite(),
        "Forsterite": minerals.SLB_2011.forsterite(),
        "Fayalite": minerals.SLB_2011.fayalite(),
        "Pyrope": minerals.SLB_2011.pyrope(),
        "Almandine": minerals.SLB_2011.almandine(),
        "Phlogopite": minerals.JH_2015.phl(),
        "Orthoclase": minerals.HP_2011_ds62.hol(),
        "Hornblende": minerals.SLB_2011.mg_tschermaks()
    }


# =============================================================================
# FUNCIONES DE CONDUCTIVIDAD TÉRMICA
# =============================================================================

def lambda_lattice(mineral, T, P=0.0):
    """
    Conductividad térmica lattice (fonónica) dependiente de T y P.
    
    λ_lattice = λ₀ (298/T)ⁿ (1 + K'ₜ/Kₜ · P[GPa])
    
    Parameters
    ----------
    mineral : str
        Nombre del mineral
    T : float
        Temperatura (K)
    P : float
        Presión (Pa)
        
    Returns
    -------
    float
        Conductividad lattice (W/m·K)
    """
    p = MINERAL_PARAMS[mineral]
    P_GPa = P / 1e9
    KT = p["KT"]
    KTp = p["KTp"]
    lam0 = p["lambda0"]
    n = p["n"]
    
    return lam0 * (298.0 / T)**n * (1.0 + (KTp / KT) * P_GPa)


def lambda_radiative(mineral, T):
    """
    Conductividad térmica radiativa (fotónica) dependiente de T.
    
    λ_rad = 0.5 λ_R,max [1 + erf((T - T_R)/ω)]
    
    Parameters
    ----------
    mineral : str
        Nombre del mineral
    T : float
        Temperatura (K)
        
    Returns
    -------
    float
        Conductividad radiativa (W/m·K)
    """
    p = MINERAL_PARAMS[mineral]
    lamRmax = p["lambdaRmax"]
    
    if lamRmax == 0.0:
        return 0.0
    
    TR = p["TR"]
    omega = p["omega"]
    
    return 0.5 * lamRmax * (1.0 + math.erf((T - TR) / omega))


def lambda_effective_VRH(comp_dict, T, P=0.0):
    """
    Conductividad térmica efectiva usando promedio Voigt-Reuss-Hill.
    
    Apropiado para agregados policristalinos (Hasterok & Chapman 2011).
    
    λ_VRH = 0.5 (λ_Voigt + λ_Reuss)
    
    donde:
    - λ_Voigt = Σ fᵢ λᵢ (suma ponderada, límite superior)
    - λ_Reuss = (Σ fᵢ/λᵢ)⁻¹ (media armónica, límite inferior)
    
    Parameters
    ----------
    comp_dict : dict
        Diccionario {mineral: fracción} normalizado a suma 1.0
    T : float
        Temperatura (K)
    P : float
        Presión (Pa)
        
    Returns
    -------
    float
        Conductividad térmica efectiva (W/m·K)
    """
    lambda_voigt = 0.0
    lambda_reuss_inv = 0.0
    
    for mineral, frac in comp_dict.items():
        lam = lambda_lattice(mineral, T, P) + lambda_radiative(mineral, T)
        lam = max(lam, 1e-6)  # Evitar división por cero
        
        lambda_voigt += frac * lam
        lambda_reuss_inv += frac / lam
    
    lambda_reuss = 1.0 / lambda_reuss_inv
    lambda_vrh = 0.5 * (lambda_voigt + lambda_reuss)
    
    return lambda_vrh


# =============================================================================
# FUNCIONES DE COMPOSICIÓN Y ESTRUCTURA
# =============================================================================

def get_composition_at_depth(z, composition=None, boundaries=None):
    """
    Devuelve el diccionario de composición modal para una profundidad dada.
    
    Parameters
    ----------
    z : float
        Profundidad (m)
    composition : dict, optional
        Diccionario con claves 'upper', 'middle', 'lower', 'mantle'
    boundaries : list, optional
        Lista [d1, d2, d3] con fronteras entre capas (m)
        Default: [16e3, 23e3, 39e3] (Tierra)
        
    Returns
    -------
    dict
        Diccionario de composición modal {mineral: fracción}
    """
    if composition is None:
        composition = COMPOSITION_DEFAULT
    
    if boundaries is None:
        boundaries = [16e3, 23e3, 39e3]
    
    d1, d2, d3 = boundaries
    
    if z < d1:
        return composition["upper"]
    elif z < d2:
        return composition["middle"]
    elif z < d3:
        return composition["lower"]
    else:
        return composition["mantle"]


def normalize_modal_dict(modal_dict):
    """
    Normaliza un diccionario modal a fracciones que sumen 1.0.
    
    Parameters
    ----------
    modal_dict : dict
        Diccionario {mineral: valor}
        
    Returns
    -------
    dict
        Diccionario normalizado {mineral: fracción}
    """
    comp = {m: v for m, v in modal_dict.items() if v > 0}
    s = sum(comp.values())
    
    if s <= 0:
        raise ValueError("Suma de composiciones modal <= 0")
    
    for m in comp:
        comp[m] = comp[m] / s
    
    return comp


def modal_to_mass_fractions(modal_dict, mineral_objects, P=1e5, T=298.0):
    """
    Convierte fracciones modales (volumétricas) a fracciones de masa.
    
    Parameters
    ----------
    modal_dict : dict
        Fracciones modales {mineral: fracción_vol} (suma = 1.0)
    mineral_objects : dict
        Objetos BurnMan {mineral_name: mineral_obj}
    P : float
        Presión (Pa)
    T : float
        Temperatura (K)
        
    Returns
    -------
    dict
        Fracciones de masa {mineral: fracción_masa}
    """
    mass_props = {}
    
    for name, vol_frac in modal_dict.items():
        if name not in mineral_objects:
            raise ValueError(f"Mineral {name} no encontrado en mineral_objects.")
        
        mineral = mineral_objects[name]
        single = Composite([mineral], [1.0])
        single.set_state(P, T)
        rho = float(single.density)  # kg/m³
        
        mass_props[name] = vol_frac * rho
    
    total = sum(mass_props.values())
    if total <= 0:
        raise ValueError("Suma de propiedades de masa = 0")
    
    mass_fracs = {name: mp / total for name, mp in mass_props.items()}
    return mass_fracs


def make_composite_from_modal(modal_dict, mineral_objects, P=1e5, T=298.0):
    """
    Crea un Composite de BurnMan desde composición modal.
    
    Pipeline: modal → mass fracs → mole fracs → Composite
    
    Parameters
    ----------
    modal_dict : dict
        Composición modal (puede no estar normalizada)
    mineral_objects : dict
        Objetos BurnMan
    P : float
        Presión (Pa)
    T : float
        Temperatura (K)
        
    Returns
    -------
    burnman.Composite
        Objeto Composite listo para set_state()
    """
    # Normalizar
    modal_norm = normalize_modal_dict(modal_dict)
    
    # Modal → masa
    mass_fracs = modal_to_mass_fractions(modal_norm, mineral_objects, P=P, T=T)
    
    # Masa → molar
    minerals_list = []
    mols = []
    
    for name, mass_frac in mass_fracs.items():
        m = mineral_objects[name]
        molar_mass = getattr(m, "molar_mass", None)
        
        if molar_mass is None:
            raise AttributeError(f"Mineral {name} no tiene 'molar_mass'")
        
        mols.append(mass_frac / molar_mass)
        minerals_list.append(m)
    
    mols = np.array(mols)
    mole_fracs = (mols / mols.sum()).tolist()
    
    comp = Composite(minerals_list, mole_fracs)
    return comp


def scale_layer_boundaries(R_planet, ref_boundaries=[16e3, 23e3, 39e3],
                           R_ref=Re, max_fraction=0.5):
    """
    Escala las profundidades de fronteras de capas según el radio del planeta.
    
    Parameters
    ----------
    R_planet : float
        Radio del planeta (m)
    ref_boundaries : list
        Profundidades de referencia [d1, d2, d3] (m)
    R_ref : float
        Radio de referencia (m), default = R_Earth
    max_fraction : float
        Fracción máxima del radio permitida para fronteras
        
    Returns
    -------
    list
        Lista [d1_scaled, d2_scaled, d3_scaled] (m)
    """
    if R_planet <= 0:
        raise ValueError("R_planet debe ser > 0")
    
    if len(ref_boundaries) != 3:
        raise ValueError("ref_boundaries debe tener 3 elementos")
    
    scale = float(R_planet) / float(R_ref)
    max_depth = float(R_planet) * float(max_fraction)
    
    scaled = []
    for b in ref_boundaries:
        d = float(b) * scale
        d = max(0.0, min(d, max_depth))
        scaled.append(d)
    
    return sorted(scaled)


# =============================================================================
# MODELO DE PRODUCCIÓN RADIACTIVA
# =============================================================================

def radiogenic_heat_profile(z, boundaries=None, A_surface=2.5e-6,
                            h_r=10e3, A_mantle=1.5e-8):
    """
    Perfil de producción radiactiva con modelo EXPONENCIAL (Hasterok 2011).
    
    A(z) = A_surface · exp(-z/h_r)  para z < z_moho
    A(z) = A_mantle                  para z ≥ z_moho
    
    Parameters
    ----------
    z : array_like
        Profundidades (m)
    boundaries : list, optional
        [d1, d2, d3] fronteras de capas (m). d3 = Moho
    A_surface : float
        Producción superficial (W/m³). Default: 2.5e-6 (2.5 μW/m³)
    h_r : float
        Profundidad característica (m). Default: 10 km
    A_mantle : float
        Producción en manto (W/m³). Default: 1.5e-8
        
    Returns
    -------
    np.ndarray
        Array con A(z) en W/m³
    """
    z = np.asarray(z)
    
    if boundaries is None:
        boundaries = [16e3, 23e3, 39e3]
    
    z_moho = boundaries[2]
    
    A_out = np.zeros_like(z, dtype=float)
    
    for i, zi in enumerate(z):
        if zi < z_moho:
            A_out[i] = A_surface * np.exp(-zi / h_r)
        else:
            A_out[i] = A_mantle
    
    return A_out


# =============================================================================
# FUNCIÓN PRINCIPAL: CALCULAR PERFIL GEOTÉRMICO CON DETECCIÓN DE LÍMITES
# =============================================================================

def calculate_geotherm(rocks, q_s, z_max, dz, R_planet, M_total,
                      composition=None, boundaries=None,
                      P_top=1e5, T_top=288.0, rho_top=2800.0, g_top=None,
                      A_surface=2.5e-6, h_r=10e3, A_mantle=1.5e-8,
                      T_max_safe=2150.0,
                      max_iter_T=60, max_iter_P=60,
                      tol_T=1e-3, tol_P=1e-3,
                      DEBUG=False):
    """
    Calcula perfiles T(z), P(z), ρ(z), q(z) integrando ecuación de conducción 1D.
    
    Ecuación de conducción en estado estacionario:
    d/dz[λ(z,T,P) dT/dz] + A(z) = 0
    
    Resuelve iterativamente el acoplamiento T-P-ρ usando BurnMan para
    obtener propiedades físicas consistentes con la mineralogía.
    
    Parameters
    ----------
    rocks : dict
        Composites de BurnMan para cada capa:
        {'upper': Composite, 'middle': Composite, 'lower': Composite, 'mantle': Composite}
    q_s : float
        Flujo de calor superficial (W/m²)
    z_max : float
        Profundidad máxima de integración (m)
    dz : float
        Paso de integración (m)
    R_planet : float
        Radio del planeta (m)
    M_total : float
        Masa total del planeta (kg)
    composition : dict, optional
        Composiciones modales por capa. Default: COMPOSITION_DEFAULT
    boundaries : list, optional
        [d1, d2, d3] fronteras entre capas (m). Default: [16e3, 23e3, 39e3]
    P_top : float
        Presión superficial (Pa). Default: 1e5
    T_top : float
        Temperatura superficial (K). Default: 288
    rho_top : float
        Densidad superficial (kg/m³). Default: 2800
    g_top : float, optional
        Gravedad superficial (m/s²). Si None, se calcula desde M_total
    A_surface : float
        Producción radiactiva superficial (W/m³). Default: 2.5e-6
    h_r : float
        Profundidad característica de decaimiento (m). Default: 10e3
    A_mantle : float
        Producción radiactiva del manto (W/m³). Default: 1.5e-8
    T_max_safe : float
        Temperatura máxima segura (K) antes de límites de BurnMan. Default: 2150
        Si la temperatura calculada excede este límite, el cálculo se detiene
        y retorna el perfil parcial hasta donde fue posible calcular.
    max_iter_T, max_iter_P : int
        Número máximo de iteraciones
    tol_T, tol_P : float
        Tolerancias de convergencia
    DEBUG : bool
        Imprimir información de depuración
        
    Returns
    -------
    pd.DataFrame
        DataFrame con columnas:
        - depth_m: profundidad (m)
        - depth_km: profundidad (km)
        - T_K: temperatura (K)
        - P_Pa: presión (Pa)
        - P_GPa: presión (GPa)
        - rho_kg_m3: densidad (kg/m³)
        - q_W_m2: flujo de calor (W/m²)
        - A_uW_m3: producción radiactiva (μW/m³)
        
    Notes
    -----
    Si el cálculo se detiene anticipadamente por límites de temperatura,
    el DataFrame solo contiene las capas calculadas exitosamente (perfil parcial).
    """
    if composition is None:
        composition = COMPOSITION_DEFAULT
    
    if boundaries is None:
        boundaries = [16e3, 23e3, 39e3]
    
    d1, d2, d3 = boundaries
    
    # Crear malla
    z = np.arange(0.0, z_max + dz, dz)
    nz = z.size
    
    T = np.zeros(nz)
    q = np.zeros(nz)
    P = np.zeros(nz)
    rhos = np.zeros(nz)
    
    # Condiciones iniciales
    T[0] = T_top
    q[0] = q_s
    P[0] = P_top
    rhos[0] = rho_top
    
    # Perfil de producción radiactiva
    A = radiogenic_heat_profile(z, boundaries=boundaries,
                                A_surface=A_surface, h_r=h_r,
                                A_mantle=A_mantle)
    
    if A.shape[0] != nz:
        raise ValueError("A_profile length mismatch")
    
    # Inicializar gravedad
    mass_above = 0.0
    
    if g_top is None:
        g_surface = G * M_total / (R_planet**2)
    else:
        g_surface = g_top
    
    # Variables para detección de límites
    stopped_early = False
    last_valid_index = 0
    
    # Bucle principal por capas
    for i in range(nz - 1):
        Ai = float(A[i])
        qi = float(q[i])
        Ti = float(T[i])
        zi = float(z[i])
        Pi = float(P[i])
        rhoi = float(rhos[i])
        
        # DETECCIÓN DE LÍMITE: Si Ti ya excede el límite, detener
        if Ti > T_max_safe:
            if DEBUG or True:  # Siempre notificar cuando se detiene
                print(f"⚠️  Cálculo detenido en z = {zi/1000:.2f} km (T = {Ti:.1f} K > {T_max_safe:.1f} K)")
                print(f"   BurnMan puede fallar a temperaturas mayores.")
                print(f"   Retornando perfil parcial con {i} capas calculadas.")
            stopped_early = True
            last_valid_index = i
            break
        
        # Radio en el centro de la capa
        depth_mid = zi + dz / 2.0
        r_mid = R_planet - depth_mid
        
        if r_mid <= 0:
            raise RuntimeError(f"r_mid <= 0 en capa {i}")
        
        # Seleccionar Composite según profundidad
        if depth_mid < d1:
            burn_comp = rocks['upper']
        elif depth_mid < d2:
            burn_comp = rocks['middle']
        elif depth_mid < d3:
            burn_comp = rocks['lower']
        else:
            burn_comp = rocks['mantle']
        
        # Obtener y normalizar composición modal
        comp_modal_raw = get_composition_at_depth(depth_mid, composition, boundaries)
        comp_modal = normalize_modal_dict(comp_modal_raw)
        
        # Estimación inicial de P_mid
        P_mid = Pi + rhoi * g_surface * dz
        
        # Lambda inicial
        lam_guess = lambda_effective_VRH(comp_modal, max(Ti, 298.0), P_mid)
        if not np.isfinite(lam_guess) or lam_guess <= 0:
            raise RuntimeError(f"lambda inicial inválida en capa {i}")
        
        # Estimación inicial de T_next
        T_next = Ti + (qi / lam_guess) * dz - (Ai / (2.0 * lam_guess)) * dz * dz
        
        # Iteración anidada: T_next <-> λ(T_next, P_mid) con P_mid <-> ρ_mid
        converged_T = False
        rho_mid = rhoi
        shell_mass = 0.0
        burnman_failed = False
        
        for it_T in range(max_iter_T):
            T_mid = 0.5 * (Ti + T_next)
            
            # Verificación preventiva: si T_next se acerca al límite
            if T_next > T_max_safe:
                if DEBUG or True:
                    print(f"⚠️  T calculada ({T_next:.1f} K) excede límite seguro en z = {zi/1000:.2f} km")
                    print(f"   Deteniendo cálculo en capa {i}.")
                stopped_early = True
                last_valid_index = i
                burnman_failed = True
                break
            
            # Iterar P_mid <-> ρ_mid
            P_mid_local = P_mid
            
            for it_P in range(max_iter_P):
                try:
                    burn_comp.set_state(P_mid_local, T_mid)
                    rho_mid = float(burn_comp.density)
                except (AssertionError, Exception) as e:
                    # BurnMan falló - temperatura/presión fuera de rango
                    if DEBUG or True:
                        print(f"⚠️  BurnMan falló en z = {zi/1000:.2f} km (T_mid = {T_mid:.1f} K, P = {P_mid_local/1e9:.3f} GPa)")
                        print(f"   Error: {type(e).__name__}")
                        print(f"   Deteniendo cálculo. Retornando perfil parcial.")
                    stopped_early = True
                    last_valid_index = i
                    burnman_failed = True
                    break
                
                # Masa de la capa (shell)
                shell_mass = 4.0 * math.pi * (r_mid**2) * rho_mid * dz
                
                # Masa encerrada al radio r_mid
                M_enclosed = M_total - mass_above - 0.5 * shell_mass
                M_enclosed = max(M_enclosed, M_total * 1e-12)
                
                # Gravedad local en r_mid
                g_mid = G * M_enclosed / (r_mid**2)
                
                # Actualizar P_mid
                P_mid_new = Pi + rho_mid * g_mid * dz
                
                if abs(P_mid_new - P_mid_local) < tol_P:
                    P_mid_local = P_mid_new
                    break
                
                # Relajación
                P_mid_local = 0.5 * P_mid_local + 0.5 * P_mid_new
            
            # Si BurnMan falló en el loop de presión, salir del loop de temperatura
            if burnman_failed:
                break
            
            # Actualizar λ con P_mid_local y T_next
            lam = lambda_effective_VRH(comp_modal, max(T_next, 298.0), P_mid_local)
            
            if not np.isfinite(lam) or lam <= 0:
                raise RuntimeError(f"lambda inválida en capa {i}")
            
            # Recalcular T_new
            T_new = Ti + (qi / lam) * dz - (Ai / (2.0 * lam)) * dz * dz
            
            # DEBUG
            if DEBUG and (i < 6 or i % 500 == 0):
                print(f"capa {i} z={zi/1000:.2f} km it_T={it_T:02d} "
                      f"P_mid={P_mid_local:.3e} ρ_mid={rho_mid:.1f} "
                      f"g_mid={g_mid:.3f} λ={lam:.4e} T_new={T_new:.4f}")
            
            # Convergencia en T
            if abs(T_new - T_next) < tol_T:
                T_next = T_new
                P_mid = P_mid_local
                converged_T = True
                break
            
            # Relajación y continuar
            T_next = 0.5 * T_next + 0.5 * T_new
            P_mid = P_mid_local
        
        # Si BurnMan falló, salir del bucle principal
        if burnman_failed:
            break
        
        if not converged_T and DEBUG:
            print(f"[WARN] capa {i} no convergió en T tras {max_iter_T} iteraciones")
        
        # Actualizar masa acumulada
        mass_above += shell_mass
        
        # Asignar resultados
        T[i+1] = T_next
        q[i+1] = qi - Ai * dz
        P[i+1] = P_mid
        rhos[i+1] = rho_mid
        last_valid_index = i + 1
        
        if DEBUG and (i < 6 or i % 500 == 0):
            print(f"-> resultado capa {i}: T_bot={T[i+1]:.3f} K, "
                  f"P_bot={P[i+1]:.3e} Pa, ρ_bot={rhos[i+1]:.1f}, "
                  f"mass_above={mass_above:.3e}")
    
    # Si se detuvo anticipadamente, truncar arrays a las capas válidas
    if stopped_early:
        z = z[:last_valid_index+1]
        T = T[:last_valid_index+1]
        P = P[:last_valid_index+1]
        rhos = rhos[:last_valid_index+1]
        q = q[:last_valid_index+1]
        A = A[:last_valid_index+1]
        
        print(f"\n📊 Perfil parcial retornado:")
        print(f"   Profundidad calculada: 0 - {z[-1]/1000:.2f} km (de {z_max/1000:.1f} km solicitados)")
        print(f"   Temperatura máxima alcanzada: {T[-1]:.1f} K")
        print(f"   Capas calculadas: {last_valid_index+1} / {nz}")
        print(f"   Porcentaje completado: {100*(last_valid_index+1)/nz:.1f}%\n")
    
    # Crear DataFrame de resultados
    df = pd.DataFrame({
        "depth_m": z,
        "depth_km": z / 1000.0,
        "T_K": T,
        "P_Pa": P,
        "P_GPa": P / 1e9,
        "rho_kg_m3": rhos,
        "q_W_m2": q,
        "A_uW_m3": A * 1e6
    })
    
    return df


# =============================================================================
# FUNCIÓN AUXILIAR: PREPARAR ROCKS DICT
# =============================================================================

def prepare_rocks_dict(composition=None, mineral_objects=None, P=1e5, T=288.0):
    """
    Prepara el diccionario de Composites de BurnMan para cada capa.
    
    Parameters
    ----------
    composition : dict, optional
        Composiciones modales. Default: COMPOSITION_DEFAULT
    mineral_objects : dict, optional
        Objetos BurnMan. Si None, se crean con get_mineral_objects()
    P : float
        Presión de referencia (Pa)
    T : float
        Temperatura de referencia (K)
        
    Returns
    -------
    dict
        {'upper': Composite, 'middle': Composite, 'lower': Composite, 'mantle': Composite}
    """
    if composition is None:
        composition = COMPOSITION_DEFAULT
    
    if mineral_objects is None:
        mineral_objects = get_mineral_objects()
    
    rocks = {}
    for key in ['upper', 'middle', 'lower', 'mantle']:
        rocks[key] = make_composite_from_modal(
            composition[key], mineral_objects, P=P, T=T
        )
    
    return rocks


# =============================================================================
# MODELOS DE EVOLUCIÓN TEMPORAL
# =============================================================================

def q_s_turcotte(t_Ga, q0=65e-3, tau=2.0):
    """
    Modelo de flujo de calor temporal de Turcotte & Schubert (2014).
    
    q_s(t) = q₀ · exp(t/τ)
    
    Modelo exponencial simple basado en el decaimiento radiactivo promedio.
    
    Parameters
    ----------
    t_Ga : float or array
        Tiempo antes del presente (Ga). Positivo hacia el pasado.
        Ej: t=0 (presente), t=1 (hace 1 Ga), t=4.5 (formación de la Tierra)
    q0 : float
        Flujo de calor superficial actual (W/m²). Default: 65e-3 (65 mW/m²)
    tau : float
        Escala de tiempo característica (Ga). Default: 2.0 Ga
        (tiempo de decaimiento efectivo de elementos radiactivos)
        
    Returns
    -------
    float or array
        Flujo de calor superficial (W/m²)
        
    Examples
    --------
    >>> q_s_turcotte(0.0)       # Presente
    0.065  # 65 mW/m²
    
    >>> q_s_turcotte(1.0)       # Hace 1 Ga
    0.107  # ~107 mW/m²
    
    >>> q_s_turcotte(4.5)       # Formación de la Tierra
    0.778  # ~778 mW/m²
    
    References
    ----------
    Turcotte, D. L., & Schubert, G. (2014). Geodynamics (3rd ed.).
    Cambridge University Press.
    """
    return q0 * np.exp(t_Ga / tau)


def A_surface_temporal(t_Ga, A0=2.5e-6, tau=2.0):
    """
    Producción radiactiva superficial en función del tiempo.
    
    A_surface(t) = A₀ · exp(t/τ)
    
    Parameters
    ----------
    t_Ga : float or array
        Tiempo antes del presente (Ga)
    A0 : float
        Producción radiactiva actual (W/m³). Default: 2.5e-6 (2.5 μW/m³)
    tau : float
        Escala de tiempo del decaimiento (Ga). Default: 2.0 Ga
        
    Returns
    -------
    float or array
        Producción radiactiva superficial (W/m³)
        
    Notes
    -----
    - Sigue el mismo decaimiento exponencial que el flujo de calor
    - En el pasado había más elementos radiactivos
    - Se usa en el perfil: A(z,t) = A_surface(t) · exp(-z/h_r)
    """
    return A0 * np.exp(t_Ga / tau)


def calculate_geotherm_evolution(rocks, composition,
                                 R_planet=Re, M_total=Me,
                                 z_max=100e3, dz=100.0,
                                 boundaries=None,
                                 T_top=288.0, h_r=10e3,
                                 q0=65e-3, tau=2.0, t_Ga=np.linspace(0.001, 2.5, 10),
                                 T_max_safe=2150.0,):
    """
    Calcula la evolución temporal del gradiente geotérmico.
    
    Usa el modelo de Turcotte & Schubert (2014) para el flujo de calor
    temporal y calcula perfiles geotérmicos para múltiples épocas.
    
    Parameters
    ----------
    rocks : dict
        Composites de BurnMan {'upper', 'middle', 'lower', 'mantle'}
    composition : dict
        Composiciones modales por capa
    R_planet : float
        Radio del planeta (m). Default: R_Earth
    M_total : float
        Masa del planeta (kg). Default: M_Earth
    z_max : float
        Profundidad máxima (m). Default: 100 km
    dz : float
        Paso de profundidad (m). Default: 100 m
    boundaries : list, optional
        [d1, d2, d3] fronteras de capas (m)
    T_top : float
        Temperatura superficial (K). Default: 288
    h_r : float
        Profundidad característica radiactiva (m). Default: 10 km
    q0 : float
        Flujo de calor actual (W/m²). Default: 65e-3 (65 mW/m²)
    tau : float
        Escala de tiempo (Ga). Default: 2.0
    T_max_safe : float
        Temperatura máxima segura (K). Default: 2150
    n_times : int
        Número de tiempos a calcular. Default: 10
    t_range : tuple
        (t_min, t_max) en Ga. Default: (0.001, 2.0)
        
    Returns
    -------
    dict
        Diccionario con:
        - 't_Ga': array de tiempos (Ga)
        - 'q_s': array de flujos de calor (W/m²)
        - 'profiles': lista de DataFrames con perfiles T(z)
        - 'gradients': array de gradientes superficiales (K/km)
        
    Notes
    -----
    Para cada tiempo t:
    1. Calcula q_s(t) con modelo de Turcotte
    2. Calcula A_surface(t)
    3. Calcula perfil T(z) con calculate_geotherm()
    4. Extrae gradiente superficial (primeros 1 km)
    
    Si un perfil se detiene por límites de BurnMan (T > T_max_safe),
    retorna perfil parcial para ese tiempo.
    
    Examples
    --------
    >>> results = calculate_geotherm_evolution(
    ...     rocks=rocks,
    ...     composition=composition,
    ...     n_times=10
    ... )
    >>> plot_gradient_evolution(results)
    """
    if boundaries is None:
        boundaries = [16e3, 23e3, 39e3]
        
    # Calcular flujos de calor con modelo de Turcotte
    q_s = q_s_turcotte(t_Ga, q0=q0, tau=tau)
    
    # Calcular perfiles para cada tiempo
    profiles = []
    gradients = []
    
    print("=" * 80)
    print(f"CALCULANDO EVOLUCIÓN TEMPORAL DEL GRADIENTE GEOTÉRMICO")
    print(f"Modelo: Turcotte & Schubert (2014)")
    print(f"Número de tiempos: {len(t_Ga)}")
    print(f"R_planet = {R_planet/1e6:.3f} x 10^6 m")
    print(f"M_total = {M_total/1e24:.3f} x 10^24 kg")
    print("=" * 80)
    
    for i, (t, q) in enumerate(zip(t_Ga, q_s)):
        # Producción radiactiva para este tiempo
        A_surf = A_surface_temporal(t, tau=tau)
        
        # Calcular perfil geotérmico
        df = calculate_geotherm(
            rocks=rocks,
            q_s=q,
            z_max=z_max,
            dz=dz,
            R_planet=R_planet,
            M_total=M_total,
            composition=composition,
            boundaries=boundaries,
            T_top=T_top,
            A_surface=A_surf,
            h_r=h_r,
            T_max_safe=T_max_safe,
            DEBUG=False
        )
        
        profiles.append(df)
        
        # Calcular gradiente superficial (primeros 1 km)
        idx_1km = np.argmin(np.abs(df['depth_km'] - 1.0))
        dT = df['T_K'].iloc[idx_1km] - df['T_K'].iloc[0]
        dz_km = df['depth_km'].iloc[idx_1km]
        gradient = dT / dz_km  # K/km
        gradients.append(gradient)
        
        print(f"  t = {t:6.3f} Ga | q_s = {q*1000:6.1f} mW/m² | dT/dz = {gradient:5.1f} K/km")
    
    print("=" * 80)
    print()
    
    return {
        't_Ga': t_Ga,
        'q_s': q_s,
        'profiles': profiles,
        'gradients': np.array(gradients)
    }


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

# =============================================================================
# RESUMEN DE FUNCIONES DISPONIBLES
# =============================================================================
"""
FUNCIONES PRINCIPALES:
=====================

Cálculo de Perfiles:
-------------------
calculate_geotherm()              - Calcula perfil T(z), P(z), ρ(z) para un tiempo
                                   CON detección automática de límites de BurnMan

Evolución Temporal:
------------------
calculate_geotherm_evolution()    - Evolución del gradiente geotérmico en el tiempo
q_s_turcotte()                    - Modelo de flujo de calor temporal
A_surface_temporal()              - Producción radiactiva temporal

Preparación de Datos:
--------------------
get_mineral_objects()             - Obtiene objetos minerales de BurnMan
prepare_rocks_dict()              - Crea Composites por capa desde composiciones modales
make_composite_from_modal()       - Convierte composición modal → Composite

Utilidades:
----------
normalize_modal_dict()            - Normaliza composición modal a suma = 1.0
get_composition_at_depth()        - Obtiene composición según profundidad
scale_layer_boundaries()          - Escala fronteras de capas según radio del planeta
radiogenic_heat_profile()         - Perfil A(z) exponencial (Hasterok 2011)

Conductividad Térmica:
---------------------
lambda_lattice()                  - Conductividad fonónica λ(T, P)
lambda_radiative()                - Conductividad fotónica λ(T)
lambda_effective_VRH()            - Conductividad efectiva (Voigt-Reuss-Hill)

Conversiones:
------------
modal_to_mass_fractions()         - Modal (vol) → fracciones de masa
"""


if __name__ == "__main__":
    print("=" * 70)
    print("EJEMPLO DE USO: geotherm_calculator.py")
    print("=" * 70)
    
    # Preparar minerales y composites
    print("\n1. Preparando composiciones minerales...")
    mineral_objects = get_mineral_objects()
    rocks = prepare_rocks_dict(mineral_objects=mineral_objects)
    print(f"   ✓ Composites creados: {list(rocks.keys())}")
    
    # Calcular perfil geotérmico para la Tierra
    print("\n2. Calculando perfil geotérmico de la Tierra (presente)...")
    df_geotherm = calculate_geotherm(
        rocks=rocks,
        q_s=65e-3,           # 65 mW/m²
        z_max=300e3,         # 300 km
        dz=100.0,            # 100 m
        R_planet=Re,
        M_total=Me,
        boundaries=[16e3, 23e3, 39e3],
        T_top=288.0,
        P_top=1e5,
        A_surface=2.5e-6,    # 2.5 μW/m³
        h_r=10e3,            # 10 km
        A_mantle=0.015e-6,   # 0.015 μW/m³
        T_max_safe=2150.0,   # Límite de seguridad de BurnMan
        DEBUG=False
    )
    
    print(f"   ✓ Perfil calculado: {len(df_geotherm)} puntos")
    print(f"   Rango de T: {df_geotherm['T_K'].min():.1f} - {df_geotherm['T_K'].max():.1f} K")
    print(f"   Rango de P: {df_geotherm['P_Pa'].min()/1e9:.3f} - {df_geotherm['P_Pa'].max()/1e9:.3f} GPa")
    print(f"   Rango de ρ: {df_geotherm['rho_kg_m3'].min():.1f} - {df_geotherm['rho_kg_m3'].max():.1f} kg/m³")
    
    # Temperatura al Moho
    idx_moho = np.argmin(np.abs(df_geotherm['depth_km'].values - 39.0))
    T_moho = df_geotherm['T_K'].iloc[idx_moho]
    print(f"\n3. Temperatura al Moho (39 km): {T_moho:.1f} K ({T_moho-273:.1f} °C)")
    
    # Gradiente superficial
    depths_10km = df_geotherm['depth_km'].values <= 10.0
    T_10km = df_geotherm[depths_10km]['T_K'].iloc[-1]
    grad_surf = (T_10km - df_geotherm['T_K'].iloc[0]) / 10.0
    print(f"\n4. Gradiente superficial (0-10 km): {grad_surf:.1f} K/km")
    print(f"   Esperado continental: 25-30 K/km")
    
    # LAB (Litosfera-Astenosfera Boundary)
    print("\n5. Buscando LAB (Límite Litosfera-Astenosfera)...")
    for T_lab in [1350, 1450, 1573]:
        idx = np.where(df_geotherm['T_K'].values >= T_lab)[0]
        if len(idx) > 0:
            lab_depth = df_geotherm['depth_km'].iloc[idx[0]]
            print(f"   LAB a T={T_lab}K: {lab_depth:.1f} km")
    
    print("\n" + "=" * 70)
    print("EJEMPLO ADICIONAL: Evolución temporal del gradiente geotérmico")
    print("=" * 70)
    
    # Calcular evolución temporal (0-1 Ga para evitar problemas con BurnMan)
    print("\n6. Calculando evolución temporal del gradiente (0-1 Ga)...")
    
    # Nota: Usar composition desde COMPOSITION_DEFAULT
    results = calculate_geotherm_evolution(
        rocks=rocks,
        composition=COMPOSITION_DEFAULT,
        R_planet=Re,
        M_total=Me,
        z_max=100e3,
        dz=100.0,
        boundaries=[16e3, 23e3, 39e3],
        T_top=288.0,
        h_r=10e3,
        q0=65e-3,
        tau=2.0,
        T_max_safe=2150.0,
        n_times=5,
        t_range=(0.001, 1.0)
    )
    
    print("\n   Resultados de evolución temporal:")
    print(f"   Tiempos calculados: {len(results['t_Ga'])}")
    print(f"   Flujo de calor: {results['q_s'][0]*1000:.1f} - {results['q_s'][-1]*1000:.1f} mW/m²")
    print(f"   Gradiente: {results['gradients'][0]:.1f} - {results['gradients'][-1]:.1f} K/km")
    
    print("\n" + "=" * 70)
    print("✅ Módulo geotherm_calculator.py funcional")
    print("   - Detección automática de límites de BurnMan")
    print("   - Perfiles parciales cuando T > T_max_safe")
    print("   - Modelos de evolución temporal incluidos")
    print("=" * 70)
