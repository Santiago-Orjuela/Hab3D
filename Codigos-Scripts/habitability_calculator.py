"""
Módulo para calcular zonas de aguabilidad y habitabilidad subsuperficial

Este módulo implementa análisis termodinámico y biológico para identificar
regiones habitables en el subsuelo planetario basándose en perfiles geotérmicos.

Características principales:
- Cálculo de temperatura de equilibrio planetario (balance radiativo)
- Análisis de fases del agua usando estándares IAPWS (R14-08, IAPWS-95)
- Identificación de zonas de aguabilidad (agua líquida termodinámica)
- Identificación de zonas habitables (límites biológicos de extremófilos)
- Visualización de diagramas P-T, P-depth, T-depth con zonas marcadas

Referencias Científicas:
------------------------
Límites Biológicos:
- Temperatura máxima (394 K / 121°C): Kashefi & Lovley (2003), Science 301(5635), 934
  Organismo: Geogemma barossii (archaea hipertermófila)
- Presión máxima (110 MPa / 1100 bar): 
  * Yayanos et al. (1981), PNAS 78(9), 5212-5215 (Shewanella benthica)
  * Bartlett (2002), Biochimica et Biophysica Acta 1595(1-2), 367-381
  * Oger & Jebbar (2010), Research in Microbiology 161(10), 799-809

Termodinámica del Agua:
- IAPWS R14-08 (2011): Curvas de fusión de hielos Ih, III, V, VI, VII
- IAPWS-95: Ecuación de estado del agua (ebullición)
- IAPWS-08: Propiedades del agua de mar (salinidad)

Zona Habitable Circunestelar:
- Kopparapu et al. (2013): Límites Recent Venus y Early Mars
- Kopparapu et al. (2016): Actualización para estrellas F, G, K, M

Vida Subsuperficial:
- Onstott et al. (2006), Geomicrobiology Journal 23(6), 369-414



Autor: Santiago Orjuela
Fecha: Diciembre 2025
Última actualización: Diciembre 13, 2025
Basado en: IAPWS standards, Kashefi & Lovley (2003), Kopparapu et al. (2013)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geotherm_calculator as gc
from scipy.optimize import brentq
from astropy import constants as const


# Importar librerías IAPWS para propiedades del agua
try:
    from iapws import IAPWS95
    from iapws.iapws08 import _Tf, _Tb
    from iapws._iapws import _Melting_Pressure
    IAPWS_AVAILABLE = True
except ImportError:
    IAPWS_AVAILABLE = False
    print("WARNING: IAPWS library not available. Water phase calculations will be limited.")

# =============================================================================
# PHYSICAL CONSTANTS / CONSTANTES FÍSICAS
# =============================================================================
SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann constant (W m^-2 K^-4) / Constante de Stefan-Boltzmann
AU = 1.495978707e11  # Astronomical Unit (m) / Unidad Astronómica (m)
Me = const.M_earth.value # Earth mass (kg) / Masa de la Tierra (kg)
Re = const.R_earth.value # Earth radius (m) / Radio de la Tierra (m)

# BIOLOGICAL LIMITS / LÍMITES BIOLÓGICOS
T_BIO_MAX_DEFAULT = 394.0  # K (121°C) - Geogemma barossii (Kashefi & Lovley 2003)
P_BIO_MAX_DEFAULT = 110.0e6  # Pa (110 MPa, 1100 bar) - Shewanella benthica (Yayanos et al. 1981)

# WATER PHASE CONSTANTS / CONSTANTES DE FASES DEL AGUA
T_TRIPLE = 273.16  # K - Triple point temperature / Temperatura del punto triple
P_TRIPLE = 611.657  # Pa - Triple point pressure / Presión del punto triple
T_CRIT = 647.096  # K - Critical point temperature / Temperatura del punto crítico
P_CRIT = 22.064e6  # Pa (22.064 MPa) - Critical point pressure / Presión del punto crítico


# =============================================================================
# EQUILIBRIUM TEMPERATURE / TEMPERATURA DE EQUILIBRIO
# =============================================================================

def T_eq(distance_AU, L_star=1.0, albedo=0.30, tau=0.21):
    """
    Español:
    Calcula la temperatura de equilibrio planetario usando balance radiativo.
    
    La temperatura de equilibrio se obtiene igualando la energía absorbida
    de la estrella con la energía emitida por el planeta (ley de Stefan-Boltzmann).
    
    Parámetros:
        distance_AU (float): Distancia orbital en Unidades Astronómicas [UA]
        L_star (float): Luminosidad estelar normalizada a la del Sol [L_☉]
                       Default: 1.0 (Sol)
        albedo (float): Albedo planetario (fracción de luz reflejada) [adimensional]
                       Default: 0.30 (similar a la Tierra)
        tau (float): Factor de efecto invernadero [adimensional]
                    Default: 0.21 
                    tau = 1 → atmósfera idealmente opaca
                    tau = -1 → atmósfera idealmente transparente
    
    Retorna:
        float: Temperatura de equilibrio [K]
    
    Fórmula:
        T_eq = 331.0 * [(L_star * (1 - albedo)) / ((1 + tau) * d^2)]^0.25
    
    Notas:
        - La constante 331.0 K proviene de T_☉ = 5778 K y R_☉ = 6.96e8 m
        - Para la Tierra a 1 UA: T_eq ≈ 255 K (sin atmósfera), 288 K (con atmósfera)
        - Kopparapu et al. (2013) usa modelo similar para límites de zona habitable
    
    Referencias:
        - Kopparapu et al. (2013), ApJ 765, 131
        - Kasting et al. (1993), Icarus 101, 108
    
    English:
    Calculate planetary equilibrium temperature using radiative balance.
    
    The equilibrium temperature is obtained by equating the energy absorbed
    from the star with the energy emitted by the planet (Stefan-Boltzmann law).
    
    Parameters:
        distance_AU (float): Orbital distance in Astronomical Units [AU]
        L_star (float): Stellar luminosity normalized to the Sun [L_☉]
                       Default: 1.0 (Sun)
        albedo (float): Planetary albedo (fraction of reflected light) [dimensionless]
                       Default: 0.30 (Earth-like)
        tau (float): Greenhouse effect factor [dimensionless]
                    Default: 0.21 (equivalent to ~33 K warming)
                    tau = 0 → no atmosphere
                    tau > 0 → with greenhouse effect
    
    Returns:
        float: Equilibrium temperature [K]
    
    Formula:
        T_eq = 331.0 * [(L_star * (1 - albedo)) / ((1 + tau) * d^2)]^0.25
    
    Notes:
        - Constant 331.0 K comes from T_☉ = 5778 K and R_☉ = 6.96e8 m
        - For Earth at 1 AU: T_eq ≈ 255 K (no atmosphere), 288 K (with atmosphere)
        - Kopparapu et al. (2013) uses similar model for habitable zone boundaries
    
    References:
        - Kopparapu et al. (2013), ApJ 765, 131
        - Kasting et al. (1993), Icarus 101, 108
    """
    T_eq = 331.0 * ((L_star * (1 - albedo)) / ((1 + tau) * distance_AU**2))**0.25
    return T_eq


# =============================================================================
# ICE POLYMORPH SELECTION / SELECCIÓN DE POLIMORFO DE HIELO
# =============================================================================

def get_stable_ice(P_MPa):
    """
    Español:
    Determina el polimorfo de hielo estable a una presión dada.
    
    Parámetros:
        P_MPa (float): Presión en megapascales [MPa]
    
    Retorna:
        str: Tipo de hielo estable ('Ih', 'III', 'V', 'VI', 'VII')
    
    Rangos de estabilidad:
        - Ih:  0 - 209.9 MPa (hielo hexagonal ordinario)
        - III: 209.9 - 350.1 MPa (hielo tetragonal)
        - V:   350.1 - 632.4 MPa (hielo monoclínico)
        - VI:  632.4 - 2216 MPa (hielo tetragonal)
        - VII: > 2216 MPa (hielo cúbico de alta presión)
    
    Notas:
        - En la Tierra, el hielo Ih es el único encontrado naturalmente en la superficie
        - Hielos III-VII existen en el manto de lunas heladas (Europa, Encélado)
        - Estos límites son aproximados y dependen ligeramente de la temperatura
    
    Referencias:
        - IAPWS R14-08 (2011): "Revised Release on the Pressure along the 
          Melting and Sublimation Curves of Ordinary Water Substance"
    
    English:
    Determine the stable ice polymorph at a given pressure.
    
    Parameters:
        P_MPa (float): Pressure in megapascals [MPa]
    
    Returns:
        str: Stable ice type ('Ih', 'III', 'V', 'VI', 'VII')
    
    Stability ranges:
        - Ih:  0 - 209.9 MPa (ordinary hexagonal ice)
        - III: 209.9 - 350.1 MPa (tetragonal ice)
        - V:   350.1 - 632.4 MPa (monoclinic ice)
        - VI:  632.4 - 2216 MPa (tetragonal ice)
        - VII: > 2216 MPa (high-pressure cubic ice)
    
    Notes:
        - On Earth, ice Ih is the only one naturally found at the surface
        - Ices III-VII exist in the mantle of icy moons (Europa, Enceladus)
        - These boundaries are approximate and depend slightly on temperature
    
    References:
        - IAPWS R14-08 (2011): "Revised Release on the Pressure along the 
          Melting and Sublimation Curves of Ordinary Water Substance"
    """
    if P_MPa < 209.9:
        return "Ih"
    elif P_MPa < 350.1:
        return "III"
    elif P_MPa < 632.4:
        return "V"
    elif P_MPa < 2216:
        return "VI"
    else:
        return "VII"


# =============================================================================
# WATER PHASE BOUNDARIES / FRONTERAS DE FASES DEL AGUA
# =============================================================================

def _Melting_Pressure_inverse(P_MPa, ice='Ih'):
    """
    Español:
    Invierte la función _Melting_Pressure(T) para obtener T(P) dado un tipo de hielo.
    
    Usa el método de brentq (búsqueda de raíces) para resolver:
    _Melting_Pressure(T) = P_MPa
    
    Parámetros:
        P_MPa (float): Presión en megapascales [MPa]
        ice (str): Tipo de hielo ('Ih', 'III', 'V', 'VI', 'VII')
                  Default: 'Ih'
    
    Retorna:
        float: Temperatura de fusión [K]
               np.nan si no se puede resolver
    
    Notas:
        - Los rangos de búsqueda son específicos para cada polimorfo
        - Si la presión está fuera del rango de estabilidad, retorna np.nan
        - Requiere que IAPWS esté instalado
    
    English:
    Invert _Melting_Pressure(T) function to get T(P) for a given ice type.
    
    Uses brentq method (root finding) to solve:
    _Melting_Pressure(T) = P_MPa
    
    Parameters:
        P_MPa (float): Pressure in megapascals [MPa]
        ice (str): Ice type ('Ih', 'III', 'V', 'VI', 'VII')
                  Default: 'Ih'
    
    Returns:
        float: Melting temperature [K]
               np.nan if cannot be solved
    
    Notes:
        - Search ranges are specific for each polymorph
        - If pressure is outside stability range, returns np.nan
        - Requires IAPWS to be installed
    """
    if not IAPWS_AVAILABLE:
        print("ERROR: IAPWS library required for melting pressure inversion")
        return np.nan
    
    # Temperature ranges for each ice polymorph
    # Rangos de temperatura para cada polimorfo
    T_ranges = {
        'Ih':  (251.165, 273.16),
        'III': (251.166, 256.164),
        'V':   (256.165, 273.31),
        'VI':  (273.32, 355),
        'VII': (355.1, 715),
    }

    T_min, T_max = T_ranges.get(ice, (200, 800))

    try:
        return brentq(lambda T: _Melting_Pressure(T, ice=ice) - P_MPa,
                      T_min, T_max, xtol=1e-6)
    except:
        return np.nan


def T_melting_IAPWS(P, salinity=0.0, ice_type='Ih'):
    """
    Español:
    Calcula la temperatura de fusión del hielo usando estándares IAPWS.
    
    Para agua pura (salinity=0), usa IAPWS R14-08 (2011) que define las
    curvas de fusión para diferentes polimorfos de hielo.
    Para agua salada (salinity>0), usa IAPWS-08 Seawater.
    
    Parámetros:
        P (float or array): Presión [Pa]
        salinity (float): Salinidad [kg_salt/kg_water]
                         Default: 0.0 (agua pura)
        ice_type (str or None): Tipo de hielo forzado ('Ih', 'III', 'V', 'VI', 'VII')
                               Si None, selecciona automáticamente según presión
                               Default: 'Ih'
    
    Retorna:
        float or array: Temperatura de fusión [K]
    
    Notas:
        - Para P < 209.9 MPa (corteza/litosfera), hielo Ih es dominante
        - Para profundidades mayores, considerar otros polimorfos
        - La salinidad reduce la temperatura de fusión (depresión crioscópica)
    
    Referencias:
        - IAPWS R14-08 (2011): Curvas de fusión
        - IAPWS-08: Propiedades del agua de mar
    
    English:
    Calculate ice melting temperature using IAPWS standards.
    
    For pure water (salinity=0), uses IAPWS R14-08 (2011) which defines
    melting curves for different ice polymorphs.
    For saline water (salinity>0), uses IAPWS-08 Seawater.
    
    Parameters:
        P (float or array): Pressure [Pa]
        salinity (float): Salinity [kg_salt/kg_water]
                         Default: 0.0 (pure water)
        ice_type (str or None): Forced ice type ('Ih', 'III', 'V', 'VI', 'VII')
                               If None, automatically selects based on pressure
                               Default: 'Ih'
    
    Returns:
        float or array: Melting temperature [K]
    
    Notes:
        - For P < 209.9 MPa (crust/lithosphere), ice Ih is dominant
        - For greater depths, consider other polymorphs
        - Salinity reduces melting temperature (cryoscopic depression)
    
    References:
        - IAPWS R14-08 (2011): Melting curves
        - IAPWS-08: Seawater properties
    """
    if not IAPWS_AVAILABLE:
        print("ERROR: IAPWS library required for melting temperature calculation")
        return np.nan
    
    P_MPa = np.atleast_1d(P) / 1e6
    T_melt = np.zeros_like(P_MPa)

    for i, p in enumerate(P_MPa):
        # Select ice polymorph / Seleccionar polimorfo de hielo
        if ice_type is None:
            ice = get_stable_ice(p)
        else:
            ice = ice_type

        # Calculate melting temperature / Calcular temperatura de fusión
        if salinity == 0:
            try:
                T_melt[i] = _Melting_Pressure_inverse(p, ice=ice)
            except:
                T_melt[i] = np.nan
        else:
            try:
                T_melt[i] = _Tf(p, salinity)
            except:
                # Fallback to pure water / Alternativa: agua pura
                T_melt[i] = _Melting_Pressure_inverse(p, ice=ice)

    return T_melt[0] if np.isscalar(P) else T_melt


def T_boiling_IAPWS(P, salinity=0.0):
    """
    Español:
    Calcula la temperatura de saturación (ebullición) usando IAPWS-95.
    
    Para agua pura, usa la curva de saturación líquido-vapor de IAPWS-95.
    Para agua salada, usa IAPWS-08 Seawater (elevación ebulloscópica).
    
    Parámetros:
        P (float or array): Presión [Pa]
        salinity (float): Salinidad [kg_salt/kg_water]
                         Default: 0.0 (agua pura)
    
    Retorna:
        float or array: Temperatura de saturación [K]
    
    Notas:
        - Válido desde punto triple (611.657 Pa) hasta punto crítico (22.064 MPa)
        - Por encima del punto crítico, retorna T_crit (agua supercrítica)
        - Por debajo del punto triple, retorna np.nan (sublimación)
        - La salinidad aumenta la temperatura de ebullición
    
    Referencias:
        - IAPWS-95: "Revised Release on the IAPWS Formulation 1995 for the 
          Thermodynamic Properties of Ordinary Water Substance for General 
          and Scientific Use" (2016)
        - IAPWS-08: Release on the IAPWS Formulation 2008 for the Thermodynamic 
          Properties of Seawater
    
    English:
    Calculate saturation temperature (boiling) using IAPWS-95.
    
    For pure water, uses liquid-vapor saturation curve from IAPWS-95.
    For saline water, uses IAPWS-08 Seawater (ebullioscopic elevation).
    
    Parameters:
        P (float or array): Pressure [Pa]
        salinity (float): Salinity [kg_salt/kg_water]
                         Default: 0.0 (pure water)
    
    Returns:
        float or array: Saturation temperature [K]
    
    Notes:
        - Valid from triple point (611.657 Pa) to critical point (22.064 MPa)
        - Above critical point, returns T_crit (supercritical water)
        - Below triple point, returns np.nan (sublimation)
        - Salinity increases boiling temperature
    
    References:
        - IAPWS-95: "Revised Release on the IAPWS Formulation 1995 for the 
          Thermodynamic Properties of Ordinary Water Substance for General 
          and Scientific Use" (2016)
        - IAPWS-08: Release on the IAPWS Formulation 2008 for the Thermodynamic 
          Properties of Seawater
    """
    if not IAPWS_AVAILABLE:
        print("ERROR: IAPWS library required for boiling temperature calculation")
        return np.nan
    
    P_array = np.atleast_1d(P)
    T_sat = np.zeros_like(P_array)
    
    if salinity > 0:
        # Use IAPWS-08 for saline water / Usar IAPWS-08 para agua salada
        for i, p in enumerate(P_array):
            try:
                T_sat[i] = _Tb(p / 1e6, salinity)  # _Tb expects MPa / _Tb espera MPa
            except:
                # Fallback: pure water / Alternativa: agua pura
                if P_TRIPLE <= p <= P_CRIT:
                    T_sat[i] = IAPWS95(P=p/1e6, x=0).T  # Saturation line / Línea de saturación
                elif p > P_CRIT:
                    T_sat[i] = T_CRIT  # Supercritical / Supercrítico
                else:
                    T_sat[i] = np.nan
    else:
        # Pure water: use IAPWS95 directly / Agua pura: usar IAPWS95 directamente
        for i, p in enumerate(P_array):
            if P_TRIPLE <= p <= P_CRIT:
                try:
                    T_sat[i] = IAPWS95(P=p/1e6, x=0).T  # x=0 = saturated liquid / x=0 = líquido saturado
                except:
                    T_sat[i] = np.nan
            elif p > P_CRIT:
                T_sat[i] = T_CRIT  # Above critical point / Por encima del punto crítico
            else:
                T_sat[i] = np.nan
    
    return T_sat[0] if np.isscalar(P) else T_sat


# =============================================================================
# HABITABILITY ZONE IDENTIFICATION / IDENTIFICACIÓN DE ZONAS DE HABITABILIDAD
# =============================================================================

def find_liquid_zone(df_geotherm, salinity=0.0, information=True):
    """
    Español:
    Identifica la zona de aguabilidad (agua líquida) en un perfil geotérmico.
    
    La zona de aguabilidad se define como la región donde el agua es
    termodinámicamente estable en fase líquida:
        T_fusión(P) < T < T_ebullición(P)
    
    Parámetros:
        df_geotherm (pd.DataFrame): DataFrame con perfil geotérmico
                                   Debe contener columnas: 'depth_m', 'T_K', 'P_Pa'
        salinity (float): Salinidad [kg_salt/kg_water]
                         Default: 0.0 (agua pura)
        information (bool): Si True, imprime información sobre la zona encontrada
                          Default: True
    
    Retorna:
        dict: Diccionario con las siguientes claves:
            - 'liquid_zone': array con profundidades de la zona líquida [m]
            - 'indices': índices del DataFrame correspondientes a la zona
            - 'T_liquid': temperaturas en la zona líquida [K]
            - 'P_liquid': presiones en la zona líquida [Pa]
            - 'T_melt': temperaturas de fusión para todo el perfil [K]
            - 'T_boil': temperaturas de ebullición para todo el perfil [K]
            Si no se encuentra zona líquida, todos los valores son None
    
    Notas:
        - La zona de aguabilidad es necesaria pero no suficiente para habitabilidad
        - Considera efectos de salinidad si se especifica
        - Usa estándares IAPWS para todas las curvas de fase
    
    English:
    Identify the aguability zone (liquid water) in a geothermal profile.
    
    The aguability zone is defined as the region where water is
    thermodynamically stable in liquid phase:
        T_fusion(P) < T < T_boiling(P)
    
    Parameters:
        df_geotherm (pd.DataFrame): DataFrame with geothermal profile
                                   Must contain columns: 'depth_m', 'T_K', 'P_Pa'
        salinity (float): Salinity [kg_salt/kg_water]
                         Default: 0.0 (pure water)
        information (bool): If True, prints information about found zone
                          Default: True
    
    Returns:
        dict: Dictionary with the following keys:
            - 'liquid_zone': array with depths of liquid zone [m]
            - 'indices': DataFrame indices corresponding to the zone
            - 'T_liquid': temperatures in liquid zone [K]
            - 'P_liquid': pressures in liquid zone [Pa]
            - 'T_melt': melting temperatures for entire profile [K]
            - 'T_boil': boiling temperatures for entire profile [K]
            If no liquid zone is found, all values are None
    
    Notes:
        - Aguability zone is necessary but not sufficient for habitability
        - Considers salinity effects if specified
        - Uses IAPWS standards for all phase curves
    """
    z = df_geotherm['depth_m'].values
    T = df_geotherm['T_K'].values
    P = df_geotherm['P_Pa'].values

    # Calculate phase boundaries / Calcular fronteras de fases
    T_melt = T_melting_IAPWS(P, salinity=salinity)
    T_boil = T_boiling_IAPWS(P, salinity=salinity)

    # Identify liquid region / Identificar región líquida
    liquid_mask = (T > T_melt) & (T < T_boil) & ~np.isnan(T_melt)

    indices = np.where(liquid_mask)[0]
    if len(indices) == 0:
        if information:
            print("No se encontró zona de agua líquida.")
        return {
            'liquid_zone': None,
            'indices': None,
            'T_liquid': None,
            'P_liquid': None,
            'T_melt': T_melt,
            'T_boil': T_boil,
        }

    return {
        'liquid_zone': z[indices],
        'indices': indices,
        'T_liquid': T[indices],
        'P_liquid': P[indices],
        'T_melt': T_melt,
        'T_boil': T_boil,
    }


def find_habitable_zone(liquid_indices, df_geotherm,
                        T_bio_max=T_BIO_MAX_DEFAULT, 
                        P_bio_max=P_BIO_MAX_DEFAULT,
                        information=True):
    """
    Español:
    Identifica la zona habitable dentro de la zona de aguabilidad.
    
    La zona habitable es un subconjunto de la zona de aguabilidad que
    además cumple con los límites biológicos de extremófilos conocidos:
        - T < T_bio_max (temperatura máxima de vida conocida)
        - P < P_bio_max (presión máxima de vida conocida)
    
    Parámetros:
        liquid_indices (array or None): Índices de la zona líquida (del find_liquid_zone)
        df_geotherm (pd.DataFrame): DataFrame con perfil geotérmico
                                   Debe contener columnas: 'depth_m', 'T_K', 'P_Pa'
        T_bio_max (float): Temperatura máxima biológica [K]
                          Default: 394.0 K (121°C, Geogemma barossii)
        P_bio_max (float): Presión máxima biológica [Pa]
                          Default: 110e6 Pa (110 MPa, Shewanella benthica)
        information (bool): Si True, imprime información sobre la zona encontrada
                          Default: True
    
    Retorna:
        dict: Diccionario con las siguientes claves:
            - 'habitable_zone': array con profundidades de la zona habitable [m]
            - 'indices': índices del DataFrame correspondientes a la zona
            - 'T_habitable': temperaturas en la zona habitable [K]
            - 'P_habitable': presiones en la zona habitable [Pa]
            Si no se encuentra zona habitable, todos los valores son None
    
    Notas:
        - Límite térmico basado en Kashefi & Lovley (2003): 394 K
        - Límite de presión basado en Yayanos et al. (1981): 110 MPa
        - La zona habitable siempre es un subconjunto de la zona de aguabilidad
    
    Referencias:
        - Kashefi & Lovley (2003), Science 301(5635), 934
        - Yayanos et al. (1981), PNAS 78(9), 5212-5215
        - Bartlett (2002), Biochimica et Biophysica Acta 1595(1-2), 367-381
        - Oger & Jebbar (2010), Research in Microbiology 161(10), 799-809
    
    English:
    Identify the habitable zone within the aguability zone.
    
    The habitable zone is a subset of the aguability zone that also
    meets the biological limits of known extremophiles:
        - T < T_bio_max (maximum known temperature for life)
        - P < P_bio_max (maximum known pressure for life)
    
    Parameters:
        liquid_indices (array or None): Indices of liquid zone (from find_liquid_zone)
        df_geotherm (pd.DataFrame): DataFrame with geothermal profile
                                   Must contain columns: 'depth_m', 'T_K', 'P_Pa'
        T_bio_max (float): Maximum biological temperature [K]
                          Default: 394.0 K (121°C, Geogemma barossii)
        P_bio_max (float): Maximum biological pressure [Pa]
                          Default: 110e6 Pa (110 MPa, Shewanella benthica)
        information (bool): If True, prints information about found zone
                          Default: True
    
    Returns:
        dict: Dictionary with the following keys:
            - 'habitable_zone': array with depths of habitable zone [m]
            - 'indices': DataFrame indices corresponding to the zone
            - 'T_habitable': temperatures in habitable zone [K]
            - 'P_habitable': pressures in habitable zone [Pa]
            If no habitable zone is found, all values are None
    
    Notes:
        - Thermal limit based on Kashefi & Lovley (2003): 394 K
        - Pressure limit based on Yayanos et al. (1981): 110 MPa
        - Habitable zone is always a subset of aguability zone
    
    References:
        - Kashefi & Lovley (2003), Science 301(5635), 934
        - Yayanos et al. (1981), PNAS 78(9), 5212-5215
        - Bartlett (2002), Biochimica et Biophysica Acta 1595(1-2), 367-381
        - Oger & Jebbar (2010), Research in Microbiology 161(10), 799-809
    """
    z = df_geotherm['depth_m'].values
    T = df_geotherm['T_K'].values
    P = df_geotherm['P_Pa'].values

    if liquid_indices is None or len(liquid_indices) == 0:
        if information:
            print("No se encontró zona habitable porque no hay zona líquida.")
        return {
            'habitable_zone': None,
            'T_habitable': None,
            'P_habitable': None,
            'indices': None,
        }

    # Apply biological limits / Aplicar límites biológicos
    bio_mask = (T < T_bio_max) & (P < P_bio_max)

    # Intersect with liquid zone / Intersectar con zona líquida
    hab_mask = np.zeros_like(bio_mask, dtype=bool)
    hab_mask[liquid_indices] = bio_mask[liquid_indices]

    indices = np.where(hab_mask)[0]

    if len(indices) == 0:
        if information:
            print("No se encontró zona habitable.")
        return {
            'habitable_zone': None,
            'T_habitable': None,
            'P_habitable': None,
            'indices': None,
        }

    return {
        'habitable_zone': z[indices],
        'T_habitable': T[indices],
        'P_habitable': P[indices],
        'indices': indices,
    }


# =============================================================================
# ZONE SUMMARY AND REPORTING / RESUMEN E INFORME DE ZONAS
# =============================================================================

def liq_hab_zone_data(liquid_zone_data, habitable_zone_data):
    """
    Español:
    Imprime un resumen detallado de las zonas de aguabilidad y habitabilidad.
    
    Parámetros:
        liquid_zone_data (dict): Diccionario retornado por find_liquid_zone()
        habitable_zone_data (dict): Diccionario retornado por find_habitable_zone()
    
    Imprime:
        - Profundidad, temperatura y presión de la zona de aguabilidad
        - Espesor de la zona de aguabilidad
        - Profundidad, temperatura y presión de la zona habitable
        - Espesor de la zona habitable
        - Porcentaje de zona habitable respecto a zona de aguabilidad
    
    Notas:
        - Las temperaturas se muestran en K y °C
        - Las presiones se muestran en MPa
        - Las profundidades y espesores se muestran en km
    
    English:
    Print a detailed summary of aguability and habitability zones.
    
    Parameters:
        liquid_zone_data (dict): Dictionary returned by find_liquid_zone()
        habitable_zone_data (dict): Dictionary returned by find_habitable_zone()
    
    Prints:
        - Depth, temperature and pressure of aguability zone
        - Thickness of aguability zone
        - Depth, temperature and pressure of habitable zone
        - Thickness of habitable zone
        - Percentage of habitable zone relative to aguability zone
    
    Notes:
        - Temperatures shown in K and °C
        - Pressures shown in MPa
        - Depths and thicknesses shown in km
    """
    if liquid_zone_data['liquid_zone'] is not None:
        liq_top = liquid_zone_data['liquid_zone'][0]
        liq_bot = liquid_zone_data['liquid_zone'][-1]

        # Aguability Zone Summary / Resumen de Zona de Aguabilidad
        print(f"\n💧 ZONA DE AGUABILIDAD (Agua Líquida):")
        print(f"   Profundidad: {liquid_zone_data['liquid_zone'][0] / 1000:.2f} - {liquid_zone_data['liquid_zone'][-1] / 1000:.2f} km")
        print(f"   Temperatura: {liquid_zone_data['T_liquid'][0]:.2f} - {liquid_zone_data['T_liquid'][-1]:.2f} K")
        print(f"   Presión: {liquid_zone_data['P_liquid'][0]/1e6:.2f} - {liquid_zone_data['P_liquid'][-1]/1e6:.2f} MPa")
        print(f"   Espesor: {(liq_bot - liq_top)/1000:.3f} km")

        # Habitable Zone Summary / Resumen de Zona Habitable
        if habitable_zone_data['habitable_zone'] is not None:
            bio_top = habitable_zone_data['habitable_zone'][0]
            bio_bot = habitable_zone_data['habitable_zone'][-1]

            print(f"\n🦠 ZONA HABITABLE (Límites Biológicos):")
            print(f"   Profundidad: {habitable_zone_data['habitable_zone'][0] / 1000:.2f} - {habitable_zone_data['habitable_zone'][-1] / 1000:.2f} km")
            print(f"   Temperatura: {habitable_zone_data['T_habitable'][0]:.2f} - {habitable_zone_data['T_habitable'][-1]:.2f} K ({habitable_zone_data['T_habitable'][0]-273.15:.1f} - {habitable_zone_data['T_habitable'][-1]-273.15:.1f}°C)")
            print(f"   Presión: {habitable_zone_data['P_habitable'][0]/1e6:.2f} - {habitable_zone_data['P_habitable'][-1]/1e6:.2f} MPa")
            print(f"   Espesor: {(bio_bot - bio_top)/1000:.3f} km")

            # Percentage of habitable zone / Porcentaje de zona habitable
            frac_habitable = ((bio_bot - bio_top) / (liq_bot - liq_top)) * 100
            print(f"\n📊 Zona Habitable = {frac_habitable:.1f}% de la Zona de Aguabilidad")
        else:
            print(f"\n🦠 ZONA HABITABLE: No existe")

    else:
        print("No se encontró zona de agua líquida.")
        
    print("\n" + "=" * 80)


# =============================================================================
# VISUALIZATION / VISUALIZACIÓN
# =============================================================================

def plot_habitability_zones(df_geotherm, liquid_zone_data, habitable_zone_data, 
                            figsize=(25, 10), save_path=None):
    """
    Español:
    Crea visualización de tres paneles de las zonas de habitabilidad.
    
    Genera tres gráficos:
    1. Diagrama P-T: Muestra curvas de fase, zona líquida, zona habitable y perfil geotérmico
    2. Perfil P-depth: Presión vs profundidad con zonas marcadas
    3. Perfil T-depth: Temperatura vs profundidad con zonas marcadas
    
    Parámetros:
        df_geotherm (pd.DataFrame): DataFrame con perfil geotérmico
        liquid_zone_data (dict): Diccionario retornado por find_liquid_zone()
        habitable_zone_data (dict): Diccionario retornado por find_habitable_zone()
        figsize (tuple): Tamaño de la figura (ancho, alto) en pulgadas
                        Default: (25, 10)
        save_path (str or None): Ruta para guardar la figura
                                Si None, solo muestra la figura
                                Default: None
    
    Retorna:
        matplotlib.figure.Figure: Objeto figura de matplotlib
    
    Notas:
        - Las zonas se marcan con colores: azul (aguabilidad), verde (habitable)
        - Los límites biológicos se muestran como líneas discontinuas
        - Las curvas de fase IAPWS se muestran en el diagrama P-T
    
    English:
    Create three-panel visualization of habitability zones.
    
    Generates three plots:
    1. P-T Diagram: Shows phase curves, liquid zone, habitable zone and geothermal profile
    2. P-depth Profile: Pressure vs depth with marked zones
    3. T-depth Profile: Temperature vs depth with marked zones
    
    Parameters:
        df_geotherm (pd.DataFrame): DataFrame with geothermal profile
        liquid_zone_data (dict): Dictionary returned by find_liquid_zone()
        habitable_zone_data (dict): Dictionary returned by find_habitable_zone()
        figsize (tuple): Figure size (width, height) in inches
                        Default: (25, 10)
        save_path (str or None): Path to save the figure
                                If None, only displays the figure
                                Default: None
    
    Returns:
        matplotlib.figure.Figure: Matplotlib figure object
    
    Notes:
        - Zones are marked with colors: blue (aguability), green (habitable)
        - Biological limits shown as dashed lines
        - IAPWS phase curves shown in P-T diagram
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])  # P-T Diagram / Diagrama P-T
    ax2 = fig.add_subplot(gs[0, 1])  # P-depth / P-profundidad
    ax3 = fig.add_subplot(gs[0, 2])  # T-depth / T-profundidad

    # Extract data / Extraer datos
    z = df_geotherm['depth_km'].values
    T = df_geotherm['T_K'].values
    P = df_geotherm['P_Pa'].values / 1e6  # Convert to MPa / Convertir a MPa

    # Zone data / Datos de zonas
    if liquid_zone_data['liquid_zone'] is not None:
        z_water = liquid_zone_data['liquid_zone'] / 1000  # km
        T_water = liquid_zone_data['T_liquid']  # K
        P_water = liquid_zone_data['P_liquid'] / 1e6  # MPa
        T_melt = liquid_zone_data['T_melt']
        T_boil = liquid_zone_data['T_boil']
        T_melt_water = T_melt[:len(P_water)]
        T_boil_water = T_boil[:len(P_water)]
    else:
        # No liquid zone found / No se encontró zona líquida
        ax1.text(0.5, 0.5, 'No liquid water zone found', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        ax2.text(0.5, 0.5, 'No liquid water zone found', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax3.text(0.5, 0.5, 'No liquid water zone found', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        return fig

    if habitable_zone_data['habitable_zone'] is not None:
        z_hab = habitable_zone_data['habitable_zone'] / 1000  # km
        T_hab = habitable_zone_data['T_habitable']  # K
        P_hab = habitable_zone_data['P_habitable'] / 1e6  # MPa
    
    # Biological limits / Límites biológicos
    T_bio_max = T_BIO_MAX_DEFAULT  # K (121°C)
    P_bio_max = P_BIO_MAX_DEFAULT / 1e6  # MPa

    # =============================================================================
    # PANEL 1: P-T DIAGRAM / DIAGRAMA P-T
    # =============================================================================

    # Liquid water region / Región de agua líquida
    mask_valid_melt = ~np.isnan(T_melt_water)
    P_melt_valid = P_water[mask_valid_melt]
    T_melt_valid = T_melt_water[mask_valid_melt]
    
    T_fill = np.concatenate([T_melt_valid, T_boil_water[::-1]])
    P_fill = np.concatenate([P_melt_valid, P_water[::-1]])
    ax1.fill(T_fill, P_fill, color='lightskyblue', alpha=0.4, 
            label='Liquid Water', zorder=1)

    # Habitable region / Región habitable
    if habitable_zone_data['habitable_zone'] is not None:
        T_hab_clipped = T_hab[T_hab <= T_bio_max]
        P_hab_clipped = P_hab[:len(T_hab_clipped)]
        T_melt_hab_clipped = T_melt_valid[:len(T_hab_clipped)]
        T_fill_hab = np.concatenate([T_melt_hab_clipped, T_hab_clipped[::-1]])
        P_fill_hab = np.concatenate([P_hab_clipped, P_hab_clipped[::-1]])
        ax1.fill(T_fill_hab, P_fill_hab, color='lightgreen', alpha=0.5,
                label='Habitable Zone', zorder=2)

    # Phase curves / Curvas de fase
    ax1.plot(T_melt_valid, P_melt_valid, 'b-', linewidth=2.5, 
            label='Fusion (IAPWS)', zorder=3)
    ax1.plot(T_boil_water, P_water, 'r-', linewidth=2.5, 
            label='Boiling (IAPWS)', zorder=3)

    # Geothermal profile / Perfil geotérmico
    ax1.plot(T, P, 'k-', linewidth=3, label='Geothermal Profile', zorder=4)
    
    # Biological limits / Límites biológicos
    ax1.axvline(T_bio_max, color='orange', linestyle='--', linewidth=2.5, 
                label=f'$T_{{bio}}$ = {T_bio_max:.0f} K\n(121°C)', zorder=5)
    ax1.axhline(P_bio_max, color='purple', linestyle='--', linewidth=2.5,
                label=f'$P_{{bio}}$ = {P_bio_max:.0f} MPa\n(1100 bar)', zorder=5)

    # Configuration / Configuración
    ax1.set_xlabel('Temperature [K]', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Pressure [MPa]', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.7)
    ax1.set_xlim(min(T)*0.9, max(T)*1.1)
    ax1.set_ylim(0, max(P_water)*1.1)

    # =============================================================================
    # PANEL 2: P-depth PROFILE / PERFIL P-profundidad
    # =============================================================================

    # Liquid water zone / Zona de agua líquida
    ax2.axhspan(z_water[0], z_water[-1], color='lightskyblue', alpha=0.4,
                label='Liquid Water', zorder=1)

    # Habitable zone / Zona habitable
    if habitable_zone_data['habitable_zone'] is not None:
        ax2.axhspan(z_hab[0], z_hab[-1], color='lightgreen', alpha=0.5,
                    label='Habitable Zone', zorder=2)
    
    # Pressure profile / Perfil de presión
    ax2.plot(P, z, 'purple', linewidth=3, label='P(z)', zorder=3)

    # Mark boundaries / Marcar fronteras
    if habitable_zone_data['habitable_zone'] is not None:
        ax2.axhline(z_hab[0], color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax2.axhline(z_hab[-1], color='red', linestyle=':', linewidth=2, alpha=0.7)

    # Configuration / Configuración
    ax2.set_xlabel('Pressure [MPa]', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Depth [km]', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.7)
    ax2.set_xlim(0, P_bio_max)
    ax2.set_ylim(z_water[-1]*1.1, 0)

    # =============================================================================
    # PANEL 3: T-depth PROFILE / PERFIL T-profundidad
    # =============================================================================

    # Liquid water zone / Zona de agua líquida
    ax3.axhspan(z_water[0], z_water[-1], color='lightskyblue', alpha=0.4,
                label='Liquid Water', zorder=1)

    # Habitable zone / Zona habitable
    if habitable_zone_data['habitable_zone'] is not None:
        ax3.axhspan(z_hab[0], z_hab[-1], color='lightgreen', alpha=0.5,
                    label='Habitable Zone', zorder=2)
    
    # Temperature profile / Perfil de temperatura
    ax3.plot(T, z, 'red', linewidth=3, label='T(z) Profile', zorder=3)

    # Mark boundaries / Marcar fronteras
    if habitable_zone_data['habitable_zone'] is not None:
        ax3.axhline(z_hab[0], color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax3.axhline(z_hab[-1], color='red', linestyle=':', linewidth=2, alpha=0.7)

    # Configuration / Configuración
    ax3.set_xlabel('Temperature [K]', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Depth [km]', fontsize=13, fontweight='bold')
    ax3.invert_yaxis()
    ax3.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.7)
    ax3.set_xlim(min(T), T_bio_max)
    ax3.set_ylim(z_water[-1]*1.1, 0)

    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig


# =============================================================================
# CONVENIENCE WRAPPER / FUNCIÓN ENVOLVENTE DE CONVENIENCIA
# =============================================================================

def analyze_habitability(df_geotherm, salinity=0.0, 
                         T_bio_max=T_BIO_MAX_DEFAULT,
                         P_bio_max=P_BIO_MAX_DEFAULT,
                         plot=True, verbose=True):
    """
    Español:
    Función conveniente que ejecuta análisis completo de habitabilidad.
    
    Ejecuta en secuencia:
    1. find_liquid_zone() - Identifica zona de aguabilidad
    2. find_habitable_zone() - Identifica zona habitable
    3. liq_hab_zone_data() - Imprime resumen
    4. plot_habitability_zones() - Crea visualización (opcional)
    
    Parámetros:
        df_geotherm (pd.DataFrame): DataFrame con perfil geotérmico
        salinity (float): Salinidad [kg_salt/kg_water]
                         Default: 0.0
        T_bio_max (float): Temperatura máxima biológica [K]
                          Default: 394.0 K
        P_bio_max (float): Presión máxima biológica [Pa]
                          Default: 110e6 Pa
        plot (bool): Si True, genera gráficos
                    Default: True
        verbose (bool): Si True, imprime información detallada
                       Default: True
    
    Retorna:
        tuple: (liquid_zone_data, habitable_zone_data)
    
    English:
    Convenience function that executes complete habitability analysis.
    
    Executes in sequence:
    1. find_liquid_zone() - Identify aguability zone
    2. find_habitable_zone() - Identify habitable zone
    3. liq_hab_zone_data() - Print summary
    4. plot_habitability_zones() - Create visualization (optional)
    
    Parameters:
        df_geotherm (pd.DataFrame): DataFrame with geothermal profile
        salinity (float): Salinity [kg_salt/kg_water]
                         Default: 0.0
        T_bio_max (float): Maximum biological temperature [K]
                          Default: 394.0 K
        P_bio_max (float): Maximum biological pressure [Pa]
                          Default: 110e6 Pa
        plot (bool): If True, generate plots
                    Default: True
        verbose (bool): If True, print detailed information
                       Default: True
    
    Returns:
        tuple: (liquid_zone_data, habitable_zone_data)
    """
    # Find liquid zone / Encontrar zona líquida
    liquid_zone_data = find_liquid_zone(df_geotherm, salinity=salinity, 
                                       information=verbose)
    
    # Find habitable zone / Encontrar zona habitable
    habitable_zone_data = find_habitable_zone(
        liquid_zone_data['indices'], 
        df_geotherm,
        T_bio_max=T_bio_max,
        P_bio_max=P_bio_max,
        information=verbose
    )
    
    # Print summary / Imprimir resumen
    if verbose:
        liq_hab_zone_data(liquid_zone_data, habitable_zone_data)
    
    # Plot if requested / Graficar si se solicita
    if plot and liquid_zone_data['liquid_zone'] is not None:
        plot_habitability_zones(df_geotherm, liquid_zone_data, habitable_zone_data)
    
    return liquid_zone_data, habitable_zone_data

def Volumen3D(distance,
              rocks, 
              R_planet,
              M_planet,
              qs,
              A_surface=2.5e-6,
              h_r=10e3):
    
    """Calcula el volumen de agua líquida y zona habitable en un planeta dado un perfil geotérmico 1D.  
    Parámetros:
        distance (float): Distancia del planeta a su estrella [UA]
        rocks (str): Tipo de roca para el cálculo del geotermo
        R_planet (float): Radio del planeta [m]
        M_planet (float): Masa del planeta [kg]
        qs (float): Flujo de calor superficial [W/m²]
        A_surface (float): Aceleración térmica superficial [1/K]
        h_r (float): Profundidad de escala de la aceleración térmica [m]
    Retorna:
        dict: Diccionario con los siguientes claves:
            - 'liquid_volume_km3': Volumen de agua líquida [km³]
            - 'habitable_volume_km3': Volumen de zona habitable [km³]
    """
    
    T_surf = T_eq(rocks, R_planet, M_planet, qs)

    df_geotherm = gc.calculate_geotherm(
        rocks=rocks,
        q_s=qs,           # 65 mW/m²
        z_max=20e3,         # 10 km 
        dz=100.0,           
        R_planet=R_planet,
        M_total=M_planet,
        boundaries=gc.scale_layer_boundaries(R_planet=R_planet),
        T_top=T_surf,
        A_surface=A_surface,
        h_r=h_r
    )

    liquid_zone_data = find_liquid_zone(df_geotherm, salinity=0.0, information=False)

    if liquid_zone_data['liquid_zone'] is not None:
        habitable_zone_data = find_habitable_zone(liquid_zone_data['indices'], df_geotherm, information=False)
        
        liq_top = liquid_zone_data['liquid_zone'][0]
        liq_bot = liquid_zone_data['liquid_zone'][-1]
        liq_thickness = (liq_bot - liq_top)

        volume_liquid = (4/3) * np.pi * (Re**3 - (Re - liq_thickness)**3)

        if habitable_zone_data is not None and habitable_zone_data['habitable_zone'] is not None:
            bio_top = habitable_zone_data['habitable_zone'][0]
            bio_bot = habitable_zone_data['habitable_zone'][-1]
            bio_thickness = (bio_bot - bio_top)

            volume_habitable = (4/3) * np.pi * (Re**3 - (Re - bio_thickness)**3)
        else:
            bio_thickness = 0.0
            volume_habitable = 0.0

    else:
        liq_thickness = 0.0
        bio_thickness = 0.0
        volume_liquid = 0.0
        volume_habitable = 0.0

    return {
        'liquid_volume_km3': volume_liquid / 1e9, #km3
        'habitable_volume_km3': volume_habitable / 1e9, #km3
    }

    