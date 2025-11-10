"""
Módulo para extraer información del Grid de Planetas (PlanetaryGrid)

Este módulo contiene funciones para:
- Leer archivos STRUC.dat y TEVOL.dat del grid de planetas
- Calcular propiedades planetarias (gravedad, flujo de calor, etc.)
- Procesar modelos completos de planetas según CMF e IMF

Autor: Santiago
Fecha: Octubre 2025
"""

import re
import os
import numpy as np
import pandas as pd
from astropy import constants

# =============================================================================
# CONSTANTES FÍSICAS
# =============================================================================
G = constants.G.value  # Constante gravitacional
Me = constants.M_earth.value  # Masa de la Tierra (kg)
Re = constants.R_earth.value  # Radio de la Tierra (m)


# =============================================================================
# FUNCIONES DE LECTURA DE ARCHIVOS
# =============================================================================

def parse_norm(header_lines):
    """
    Extrae el diccionario #norm={...} si existe en el header.
    
    Parameters
    ----------
    header_lines : list of str
        Líneas de encabezado del archivo
        
    Returns
    -------
    dict
        Diccionario con valores de normalización o {} si no existe
    """
    for L in header_lines:
        if L.strip().startswith('#norm='):
            txt = L.strip()[len("#norm="):]
            try:
                norm = eval(txt)
                return norm
            except Exception as e:
                print(f"No pude evaluar #norm: {e}")
    return {}


def read_struc_dat(path):
    """
    Lee archivo STRUC.dat del grid planetario.
    
    Parameters
    ----------
    path : str
        Ruta al archivo STRUC.dat
        
    Returns
    -------
    pd.DataFrame
        DataFrame con columnas: ur, r, mr, rho, P, g, phi, T, composition
        Atributos adicionales:
        - df.attrs['header']: líneas de encabezado
        - df.attrs['norm']: diccionario de normalización
        - df.attrs['composition']: fracciones de capas
    """
    header = []
    data_lines = []
    layers = {}
    
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header.append(line.rstrip("\n"))
                # Extraer información de capas
                if line.startswith('#layer'):
                    parts = line.split()
                    if len(parts) >= 3:
                        name = parts[1]
                        frac = float(parts[2])
                        layers[name] = frac
            elif line.strip() != '':
                data_lines.append(line)
    
    # Cargar datos numéricos
    data = np.loadtxt(data_lines)
    cols = ['ur', 'r', 'mr', 'rho', 'P', 'g', 'phi', 'T', 'composition']
    df = pd.DataFrame(data, columns=cols)
    
    # Agregar metadatos como atributos
    df.attrs['header'] = header
    df.attrs['norm'] = parse_norm(header)
    df.attrs['composition'] = layers
    
    return df


def read_tevol_dat(path):
    """
    Lee archivo TEVOL.dat del grid planetario.
    
    Parameters
    ----------
    path : str
        Ruta al archivo TEVOL.dat
        
    Returns
    -------
    pd.DataFrame
        DataFrame con datos de evolución térmica
        Columnas típicas: t, Qconv, Ri, R*, Qc, Qm, Qr, Tcmb, Tl, Tup, RiFlag, Bs
    """
    # Leer línea de encabezado
    header_line = None
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header_line = line.lstrip("\n").strip()
                break
    
    if header_line is None:
        raise Exception(f"No se encontró línea header que empiece con '#' en {path}")
    
    # Procesar nombres de columnas (remover unidades entre [])
    cols = [re.sub(r'\[.*?\]', '', c).strip() 
            for c in re.split(r'\s+', header_line) 
            if c.strip() != '']
    
    # Intentar leer con pandas
    try:
        df = pd.read_csv(path, comment='#', sep='\s+', header=None, 
                        names=cols, engine='python')
        return df
    except Exception as e:
        # Fallback: usar numpy
        data = np.genfromtxt(path, comments='#', invalid_raise=False)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != len(cols):
            raise ValueError(
                f"Lectura fallida: datos con forma {data.shape} pero "
                f"{len(cols)} columnas esperadas. Error: {e}"
            )
        df = pd.DataFrame(data, columns=cols)
        return df


# =============================================================================
# FUNCIONES DE CÁLCULO DE PROPIEDADES
# =============================================================================

def calculate_gravity_profile(path_struc):
    """
    Calcula el perfil de gravedad g(r) para un planeta.
    
    Parameters
    ----------
    path_struc : str
        Ruta al archivo STRUC.dat
        
    Returns
    -------
    np.ndarray
        Array con valores de gravedad (m/s²) en cada punto radial
    """
    data_struc = read_struc_dat(path_struc)
    rs = np.array(data_struc.r)
    mrs = np.array(data_struc.mr)
    
    # Convertir a unidades SI
    Rps = Re * rs  # m
    Mps = Me * mrs  # kg
    
    gs = np.zeros_like(rs)
    nz = rs > 0
    gs[nz] = G * Mps[nz] / Rps[nz]**2
    
    return gs


def get_radius(path_struc):
    """
    Obtiene el radio del planeta.
    
    Parameters
    ----------
    path_struc : str
        Ruta al archivo STRUC.dat
        
    Returns
    -------
    float or None
        Radio en unidades de R_Earth, o None si falla
    """
    data_struc = read_struc_dat(path_struc)
    norm = data_struc.attrs['norm']
    
    # Intentar obtener del diccionario norm
    if 'R' in norm:
        try:
            return norm['R']
        except Exception:
            pass
    
    # Fallback: último valor de la columna r
    try:
        r = data_struc.r.iloc[-1]
        return r
    except Exception:
        return None


def get_mass(path_struc):
    """
    Obtiene la masa del planeta.
    
    Parameters
    ----------
    path_struc : str
        Ruta al archivo STRUC.dat
        
    Returns
    -------
    float or None
        Masa en unidades de M_Earth, o None si falla
    """
    data_struc = read_struc_dat(path_struc)
    norm = data_struc.attrs['norm']
    
    # Intentar obtener del diccionario norm
    if 'M' in norm:
        try:
            return norm['M']
        except Exception:
            pass
    
    # Fallback: último valor de la columna mr
    try:
        m = data_struc.mr.iloc[-1]
        return m
    except Exception:
        return None


def get_surface_heat_flux(path_struc, path_tevol):
    """
    Calcula el flujo de calor superficial J_Q [W/m²].
    
    J_Q = Q_m / (4π R²)
    
    Parameters
    ----------
    path_struc : str
        Ruta al archivo STRUC.dat
    path_tevol : str
        Ruta al archivo TEVOL.dat
        
    Returns
    -------
    float
        Flujo de calor superficial en W/m²
    """
    data_struc = read_struc_dat(path_struc)
    data_tevol = read_tevol_dat(path_tevol)
    
    # Radio del planeta en metros
    Rps = np.array(data_struc.r) * Re
    R_planet = Rps[-1]
    
    # Flujo de calor del manto (última columna temporal)
    Qs = np.array(data_tevol.Qm)
    Q_final = Qs[-1]
    
    # Área superficial
    A = 4 * np.pi * R_planet**2
    
    # Flujo de calor superficial
    JQ = Q_final / A
    
    return JQ


def get_CMF_IMF(modelname):
    """
    Extrae los valores de CMF e IMF del nombre de un modelo.
    
    Parameters
    ----------
    modelname : str
        Nombre del modelo (ej: 'CMF_0.30-IMF_0.10')
        
    Returns
    -------
    tuple of (float, float) or (None, None)
        (CMF, IMF) o (None, None) si no coincide el patrón
    """
    model = re.match(r'^CMF_([0-9.+-eE]+)-IMF_([0-9.+-eE]+)$', modelname)
    if model:
        try:
            return float(model.group(1)), float(model.group(2))
        except Exception:
            return None, None
    return None, None


# =============================================================================
# FUNCIÓN PRINCIPAL: PROCESAR MODELO COMPLETO
# =============================================================================

def process_planet_model(model_folder):
    """
    Procesa todos los archivos de un modelo planetario (carpeta CMF_X-IMF_Y).
    
    Parameters
    ----------
    model_folder : str
        Ruta a la carpeta del modelo (ej: 'PlanetaryGrid/CMF_0.30-IMF_0.10')
        
    Returns
    -------
    pd.DataFrame
        DataFrame con una fila por cada masa del modelo, columnas:
        - Mp: masa del planeta [M_Earth]
        - Rp: radio del planeta [R_Earth]
        - P_surf: presión superficial [Pa]
        - rho_surf: densidad superficial [kg/m³]
        - T_surf: temperatura superficial [K]
        - g_surf: gravedad superficial [m/s²]
        - JQ: flujo de calor superficial [W/m²]
        
        Atributos del DataFrame:
        - df.attrs['cmf']: Core Mass Fraction
        - df.attrs['imf']: Ice Mass Fraction
        - df.attrs['mmf']: Mantle Mass Fraction (1 - CMF - IMF)
    """
    model_folder = os.path.abspath(model_folder)
    if not os.path.isdir(model_folder):
        raise FileNotFoundError(f"No existe carpeta: {model_folder}")
    
    # Extraer CMF e IMF del nombre de la carpeta
    cmf, imf = get_CMF_IMF(os.path.basename(model_folder))
    
    # Listar archivos
    files = sorted(os.listdir(model_folder))
    STRUC_files = [f for f in files if f.endswith('STRUC.dat')]
    TEVOL_files = [f for f in files if f.endswith('TEVOL.dat')]
    
    if len(TEVOL_files) == 0:
        raise FileNotFoundError(f"No se encontraron archivos TEVOL.dat en {model_folder}")
    
    rows = []
    _rx_M = re.compile(r'^M([0-9.+-eE]+)-')
    
    for tfile in TEVOL_files:
        TEVOL_path = os.path.join(model_folder, tfile)
        
        # Extraer masa del nombre del archivo
        base = os.path.basename(tfile)
        m_match = re.search(_rx_M, base)
        if not m_match:
            continue
        m = float(m_match.group(1))
        
        # Buscar archivo STRUC correspondiente
        matching_struc = None
        for sfile in STRUC_files:
            if sfile.startswith(f'M{m:0.2f}-') or sfile.startswith(f'M{m}-'):
                matching_struc = sfile
                break
        
        if matching_struc is None:
            print(f"Warning: No se encontró archivo STRUC.dat para M={m} en {model_folder}")
            continue
        
        STRUC_path = os.path.join(model_folder, matching_struc)
        
        # Leer datos
        data_struc = read_struc_dat(STRUC_path)
        data_tevol = read_tevol_dat(TEVOL_path)
        
        # Extraer propiedades
        Mp = get_mass(STRUC_path)
        Rp = get_radius(STRUC_path)
        
        try:
            gs = calculate_gravity_profile(STRUC_path)
            g_surf = gs[-1]
        except:
            g_surf = None
        
        try:
            P_surf = np.array(data_struc['P'])[-1]
            rho_surf = np.array(data_struc['rho'])[-1]
            T_surf = np.array(data_struc['T'])[-1]
        except:
            P_surf = None
            rho_surf = None
            T_surf = None
        
        try:
            JQ = get_surface_heat_flux(STRUC_path, TEVOL_path)
        except:
            JQ = None
        
        # Agregar fila
        rows.append({
            'Mp': Mp,
            'Rp': Rp,
            'P_surf': P_surf,
            'rho_surf': rho_surf,
            'T_surf': T_surf,
            'g_surf': g_surf,
            'JQ': JQ,
        })
    
    # Crear DataFrame
    df_out = pd.DataFrame(rows)
    df_out.attrs['cmf'] = cmf
    df_out.attrs['imf'] = imf
    df_out.attrs['mmf'] = 1 - cmf - imf if (cmf is not None and imf is not None) else None
    
    return df_out


# =============================================================================
# FUNCIÓN AUXILIAR: PROCESAR MÚLTIPLES MODELOS
# =============================================================================

def process_all_models(planetary_grid_path, imf_filter=None):
    """
    Procesa todos los modelos en el directorio PlanetaryGrid.
    
    Parameters
    ----------
    planetary_grid_path : str
        Ruta al directorio principal con carpetas CMF_X-IMF_Y
    imf_filter : float, optional
        Si se especifica, solo procesa modelos con este valor de IMF
        
    Returns
    -------
    pd.DataFrame
        DataFrame combinado con todos los modelos, incluyendo columna 'CMF'
    """
    folders = [f for f in sorted(os.listdir(planetary_grid_path)) 
               if f.startswith("CMF_")]
    
    all_data = []
    
    for folder in folders:
        cmf, imf = get_CMF_IMF(folder)
        
        # Filtrar por IMF si se especificó
        if imf_filter is not None and imf != imf_filter:
            continue
        
        folder_path = os.path.join(planetary_grid_path, folder)
        print(f"Procesando: {folder_path}")
        
        try:
            df_model = process_planet_model(folder_path)
            df_model['CMF'] = cmf
            df_model['IMF'] = imf
            all_data.append(df_model)
        except FileNotFoundError as e:
            print(f"Error: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        print("No se encontraron datos")
        return pd.DataFrame()


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EJEMPLO DE USO: planetary_grid_reader.py")
    print("=" * 70)
    
    # Ruta al grid de planetas
    planetary_grid_path = "PlanetaryGrid"
    
    # Ejemplo 1: Leer un archivo específico
    print("\n1. Leyendo archivo STRUC.dat específico...")
    path_struc = os.path.join(planetary_grid_path, "CMF_0.30-IMF_0.00", "M1.00-STRUC.dat")
    if os.path.exists(path_struc):
        data = read_struc_dat(path_struc)
        print(f"   Columnas: {list(data.columns)}")
        print(f"   Radio: {get_radius(path_struc):.3f} R_Earth")
        print(f"   Masa: {get_mass(path_struc):.3f} M_Earth")
    else:
        print(f"   Archivo no encontrado: {path_struc}")
    
    # Ejemplo 2: Procesar un modelo completo
    print("\n2. Procesando modelo completo CMF_0.30-IMF_0.00...")
    model_folder = os.path.join(planetary_grid_path, "CMF_0.30-IMF_0.00")
    if os.path.exists(model_folder):
        df_model = process_planet_model(model_folder)
        print(f"   Planetas procesados: {len(df_model)}")
        print(f"   CMF: {df_model.attrs['cmf']}")
        print(f"   IMF: {df_model.attrs['imf']}")
        print("\n   Primeros 3 planetas:")
        print(df_model.head(3))
    
    # Ejemplo 3: Procesar todos los modelos con IMF=0.00
    print("\n3. Procesando todos los modelos con IMF=0.00...")
    if os.path.exists(planetary_grid_path):
        df_all = process_all_models(planetary_grid_path, imf_filter=0.00)
        if not df_all.empty:
            print(f"   Total de planetas: {len(df_all)}")
            print(f"   Rango de masas: {df_all['Mp'].min():.2f} - {df_all['Mp'].max():.2f} M_Earth")
            print(f"   Rango de JQ: {df_all['JQ'].min()*1000:.1f} - {df_all['JQ'].max()*1000:.1f} mW/m²")
    
    print("\n" + "=" * 70)
