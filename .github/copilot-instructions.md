# PlanetaryGrid - Guía para Agentes de IA

## Descripción del Proyecto

**PlanetaryGrid** es un proyecto de grado enfocado en el modelado de estructuras interiores y evolución térmica de planetas tipo Tierra. El proyecto combina:

1. **Grid pre-computado de planetas** con diferentes composiciones (CMF: Core Mass Fraction, IMF: Ice Mass Fraction)
2. **Modelos de perfil geotérmico** para la litosfera/corteza basados en conducción térmica 1D
3. **Análisis de propiedades planetarias** (flujo de calor, estructura, campos magnéticos)

### Referencias Clave
- Anteproyecto del proyecto: `Anteproyecto.pdf`
- Bibliografía principal: carpeta `Textos/`
- Modelo geotérmico base: Hasterok & Chapman (2011)

---

## Arquitectura del Código

### 1. Scripts Principales

#### `planetary_grid_reader.py`
**Propósito**: Extraer información del grid de planetas pre-computado.

**Funciones clave**:
```python
# Lectura de archivos
read_struc_dat(path)         # Lee M*-STRUC.dat → DataFrame con estructura
read_tevol_dat(path)         # Lee M*-TEVOL.dat → DataFrame con evolución térmica

# Cálculo de propiedades
calculate_gravity_profile(path_struc)      # Perfil g(r)
get_surface_heat_flux(path_struc, path_tevol)  # J_Q = Q_m/(4πR²)
get_radius(path_struc)       # Radio del planeta [R_Earth]
get_mass(path_struc)         # Masa del planeta [M_Earth]

# Procesamiento de modelos completos
process_planet_model(model_folder)  # Procesa carpeta CMF_X-IMF_Y completa
process_all_models(grid_path, imf_filter=None)  # Procesa múltiples modelos
```

**Estructura de datos STRUC**:
- Columnas: `ur, r, mr, rho, P, g, phi, T, composition`
- `r`: radio normalizado [R_Earth]
- `mr`: masa encerrada normalizada [M_Earth]
- `rho`: densidad [kg/m³]
- `P`: presión [Pa]
- `g`: gravedad [m/s²]
- `T`: temperatura [K]
- `composition`: 0=core, 1=mantle, 2=ice

**Estructura de datos TEVOL**:
- Columnas típicas: `t, Qconv, Ri, R*, Qc, Qm, Qr, Tcmb, Tl, Tup, RiFlag, Bs`
- `t`: tiempo [yr]
- `Qm`: flujo de calor del manto [W]
- `Ri`: radio del núcleo interno [Rp]
- `Tcmb`: temperatura CMB (Core-Mantle Boundary) [K]

---

#### `geotherm_calculator.py`
**Propósito**: Calcular perfiles geotérmicos T(z), P(z), ρ(z) para la litosfera/corteza.

**Modelo físico**:
- Ecuación de conducción 1D en estado estacionario: `d/dz[λ(z,T,P) dT/dz] + A(z) = 0`
- Producción radiactiva exponencial: `A(z) = A_surface · exp(-z/h_r)` (Hasterok 2011)
- Conductividad térmica VRH (Voigt-Reuss-Hill) para agregados policristalinos
- Acoplamiento T-P-ρ usando BurnMan para propiedades dependientes de mineralogía

**Funciones clave**:
```python
# Preparación
get_mineral_objects()        # Crea objetos BurnMan para minerales
prepare_rocks_dict()         # Crea Composites por capa (upper/middle/lower/mantle)

# Modelo de conductividad térmica
lambda_lattice(mineral, T, P)      # Componente fonónica
lambda_radiative(mineral, T)       # Componente fotónica (alta T)
lambda_effective_VRH(comp_dict, T, P)  # Promedio Voigt-Reuss-Hill

# Producción radiactiva
radiogenic_heat_profile(z, A_surface=2.5e-6, h_r=10e3)  # A(z) exponencial

# Cálculo principal
calculate_geotherm(rocks, q_s, z_max, dz, R_planet, M_total, ...)
# Retorna: DataFrame con depth_m, T_K, P_Pa, rho_kg_m3, q_W_m2, A_uW_m3
```

**Parámetros importantes**:
- `q_s`: flujo de calor superficial [W/m²] (típico: 40-65 mW/m² continental)
- `A_surface`: producción radiactiva superficial [W/m³] (típico: 2.5 μW/m³)
- `h_r`: profundidad característica [m] (típico: 10 km para corteza)
- `boundaries`: fronteras de capas [d1, d2, d3] en metros (default: [16km, 23km, 39km])

---

### 2. Notebook Principal: `PlanetaryGrid.ipynb`

**Estructura del notebook** (2 partes):

**Parte 1: Extracción de datos del grid**
- Células de importación y constantes
- Funciones de lectura (`read_struc_dat`, `read_tevol_dat`)
- Cálculo de propiedades (`g(r)`, `JQ`, `table_model()`)
- Visualizaciones de perfiles (gravedad, flujo de calor)

**Parte 2: Perfil geotérmico**
- Composiciones minerales modales por capa (upper/middle/lower/mantle)
- Parámetros de conductividad térmica por mineral (MINERAL_PARAMS)
- Función `Data_perfiles()` (ahora en `geotherm_calculator.py` como `calculate_geotherm()`)
- Validación contra datos de referencia terrestres
- Aplicación a planetas del grid

**Sección final**: Markdown con documentación del modelo geotérmico
- Evolución temporal del flujo de calor (Turcotte, Korenaga, Jaupart)
- Ecuación de conducción térmica
- Modelo de conductividad VRH
- Estructura de presión con gravedad variable

---

## Convenciones y Patrones Críticos

### 1. Nomenclatura de Archivos
```
PlanetaryGrid/
├── CMF_0.XX-IMF_0.YY/          # Modelo con composición específica
│   ├── Config.py               # Configuración del modelo
│   ├── M1.00-STRUC.dat        # Estructura para masa 1.0 M_Earth
│   ├── M1.00-TEVOL.dat        # Evolución térmica para masa 1.0 M_Earth
│   └── ...
```

**Patrón de nombres**:
- CMF: Core Mass Fraction (0.10 a 0.80)
- IMF: Ice Mass Fraction (0.00 a 0.80)
- MMF: Mantle Mass Fraction = 1 - CMF - IMF (calculado, no en nombre)
- Restricción: CMF + IMF ≤ 0.90 (MMF ≥ 0.10)

### 2. Composiciones Minerales

Las composiciones se definen como **fracciones modales (volumétricas)**, no de masa:

```python
composition = {
    "upper": {"Quartz": 20, "Albite": 32, ...},   # % volumétrico
    "middle": {...},
    "lower": {...},
    "mantle": {"Forsterite": 64.04, ...}
}
```

**Pipeline de conversión**:
1. Modal (vol %) → Normalizar a suma = 1.0
2. Modal → Masa (usando ρ de BurnMan)
3. Masa → Molar (usando masa molar)
4. Molar → Composite de BurnMan

**Función**: `make_composite_from_modal(modal_dict, mineral_objects, P, T)`

### 3. Fronteras de Capas (Boundaries)

```python
boundaries = [16e3, 23e3, 39e3]  # [d1, d2, d3] en metros
# d1: upper-middle crust
# d2: middle-lower crust  
# d3: lower crust-mantle (Moho)
```

**Escalado para otros planetas**:
```python
boundaries_scaled = scale_layer_boundaries(
    R_planet, 
    ref_boundaries=[16e3, 23e3, 39e3],
    R_ref=Re,
    max_fraction=0.5  # No más profundo que R/2
)
```

### 4. Iteración Acoplada T-P-ρ

El cálculo del perfil geotérmico requiere resolver iterativamente:

```
Para cada capa i:
  1. Estimar P_mid (desde Pi y ρi-1)
  2. Estimar λ = λ(composición, Ti, P_mid)
  3. Calcular T_next con fórmula de diferencias finitas
  4. Iterar:
     a. T_mid = (Ti + T_next)/2
     b. Iterar P_mid ↔ ρ_mid:
        - set_state(P_mid, T_mid) en Composite
        - ρ_mid = Composite.density
        - g_mid = G·M(r_mid)/r_mid²
        - P_mid_new = Pi + ρ_mid·g_mid·dz
        - Repetir hasta |P_mid_new - P_mid| < tol_P
     c. λ = λ(composición, T_next, P_mid)
     d. T_new = Ti + (qi/λ)·dz - (Ai/2λ)·dz²
     e. Comprobar |T_new - T_next| < tol_T
```

**Parámetros de convergencia**:
- `max_iter_T`, `max_iter_P`: típicamente 60
- `tol_T`: 1e-3 K
- `tol_P`: 1e-3 Pa

---

## Workflows de Desarrollo

### Workflow 1: Analizar un Modelo del Grid

```python
from planetary_grid_reader import *

# Procesar modelo específico
model_path = "PlanetaryGrid/CMF_0.30-IMF_0.00"
df = process_planet_model(model_path)

# Explorar datos
print(f"CMF: {df.attrs['cmf']}, IMF: {df.attrs['imf']}")
print(df[['Mp', 'Rp', 'JQ']].head())

# Graficar flujo de calor vs masa
import matplotlib.pyplot as plt
plt.plot(df['Mp'], df['JQ']*1000)  # Convertir a mW/m²
plt.xlabel('Mass [M_Earth]')
plt.ylabel('Heat Flux [mW/m²]')
```

### Workflow 2: Calcular Perfil Geotérmico para la Tierra

```python
from geotherm_calculator import *

# Preparar composites
mineral_objects = get_mineral_objects()
rocks = prepare_rocks_dict(mineral_objects=mineral_objects)

# Calcular perfil
df_geotherm = calculate_geotherm(
    rocks=rocks,
    q_s=65e-3,           # 65 mW/m²
    z_max=300e3,         # 300 km
    dz=100.0,            # 100 m
    R_planet=Re,
    M_total=Me,
    boundaries=[16e3, 23e3, 39e3],
    T_top=288.0,
    A_surface=2.5e-6,
    h_r=10e3
)

# Validar resultados
T_moho = df_geotherm[df_geotherm['depth_km'] == 39]['T_K'].iloc[0]
print(f"T al Moho: {T_moho:.1f} K (esperado: ~900 K)")
```

### Workflow 3: Perfil Geotérmico para Planeta del Grid

```python
# Combinar ambos módulos
from planetary_grid_reader import process_planet_model, get_surface_heat_flux
from geotherm_calculator import *

# 1. Extraer datos del planeta
model_path = "PlanetaryGrid/CMF_0.30-IMF_0.10"
df_model = process_planet_model(model_path)
planet = df_model.iloc[0]  # Primer planeta

# 2. Preparar parámetros
R_planet = planet['Rp'] * Re  # m
M_total = planet['Mp'] * Me   # kg
q_surf = planet['JQ']         # W/m²
T_top = planet['T_surf']      # K
P_top = planet['P_surf']      # Pa
rho_top = planet['rho_surf']  # kg/m³

# 3. Escalar fronteras
boundaries = scale_layer_boundaries(R_planet)

# 4. Calcular perfil
mineral_objects = get_mineral_objects()
rocks = prepare_rocks_dict(mineral_objects=mineral_objects)

df_planet_geotherm = calculate_geotherm(
    rocks=rocks,
    q_s=q_surf,
    z_max=300e3,
    dz=100.0,
    R_planet=R_planet,
    M_total=M_total,
    boundaries=boundaries,
    T_top=T_top,
    P_top=P_top,
    rho_top=rho_top
)
```

---

## Trampas Comunes y Soluciones

### 1. ❌ Unidades Inconsistentes

**Problema**: Mezclar unidades normalizadas y SI.

**Solución**: 
- Grid usa unidades normalizadas: `r` en [R_Earth], `mr` en [M_Earth]
- Cálculos internos usan SI: multiplicar por `Re` y `Me`
- Flujo de calor en W/m², convertir a mW/m² para visualización

```python
# ✓ CORRECTO
R_meters = radius_normalized * Re
q_mW = heat_flux_W_m2 * 1000

# ✗ INCORRECTO
R_meters = radius_normalized  # Falta multiplicar por Re
```

### 2. ❌ Composiciones No Normalizadas

**Problema**: Pasar composiciones modales sin normalizar.

**Solución**: Usar siempre `normalize_modal_dict()`:

```python
comp_raw = {"Quartz": 20, "Albite": 30}  # Suma = 50, no 1.0
comp_norm = normalize_modal_dict(comp_raw)  # Suma = 1.0
```

### 3. ❌ BurnMan set_state() con Orden Incorrecto

**Problema**: API de BurnMan puede esperar (P,T) o (T,P) según versión.

**Solución**: Usar try-except en wrapper:

```python
try:
    composite.set_state(P, T)
except:
    composite.set_state(T, P)
```

### 4. ❌ Fronteras Físicamente Inconsistentes

**Problema**: Fronteras de capas más profundas que el radio del planeta.

**Solución**: Usar `scale_layer_boundaries()` con `max_fraction=0.5`:

```python
boundaries = scale_layer_boundaries(
    R_planet,
    max_fraction=0.5  # Limita profundidad a R/2
)
```

### 5. ❌ No Verificar Convergencia

**Problema**: Iteraciones T-P-ρ no convergen pero el código continúa.

**Solución**: Activar `DEBUG=True` para capas problemáticas:

```python
df = calculate_geotherm(..., DEBUG=True)
# Imprime detalles cada 500 capas + primeras 6 capas
```

---

## Valores de Referencia (Tierra)

### Flujo de Calor Superficial
- Continental: 40-65 mW/m²
- Oceánico: 80-100 mW/m²
- Global promedio: ~65 mW/m²

### Gradiente Geotérmico Superficial (0-10 km)
- Continental: 25-30 K/km
- Oceánico: 40-50 K/km

### Temperatura al Moho (39 km)
- Moderno: ~900 K (627 °C)
- Arcaico (3.5 Ga): ~1100-1300 K

### LAB (Litosfera-Astenosfera Boundary)
- T = 1350 K: ~80-100 km (común)
- T = 1450 K: ~100-120 km
- T = 1573 K (solidus): ~120-150 km

### Producción Radiactiva
- Superficie (corteza superior): 2.0-3.0 μW/m³
- Corteza media: 1.0-1.5 μW/m³
- Corteza inferior: 0.5-1.0 μW/m³
- Manto: 0.01-0.02 μW/m³

---

## Dependencias Críticas

```python
# Lectura/manipulación de datos
import numpy as np
import pandas as pd
import re, os, math

# Constantes físicas
from astropy import constants

# Mineralogía y ecuaciones de estado
from burnman import minerals, Composite

# Visualización (notebooks)
import matplotlib.pyplot as plt
```

**BurnMan**: Librería especializada para mineralogía y ecuaciones de estado a alta P-T.
- Provee objetos `Mineral` con propiedades termodinámicas
- `Composite` combina minerales con fracciones molares
- `set_state(P, T)` actualiza propiedades al estado (P,T)
- Acceso a `density`, `bulk_modulus`, `thermal_expansivity`, etc.

---

## Debugging y Validación

### Verificar Datos del Grid

```python
path = "PlanetaryGrid/CMF_0.30-IMF_0.00/M1.00-STRUC.dat"
df = read_struc_dat(path)

# Verificaciones básicas
assert df['r'].iloc[-1] > 0, "Radio final debe ser positivo"
assert df['mr'].iloc[-1] > 0, "Masa final debe ser positiva"
assert (df['composition'] >= 0).all(), "Composición debe ser >= 0"

# Metadatos
print(f"Norm: {df.attrs['norm']}")
print(f"Layers: {df.attrs['composition']}")
```

### Verificar Perfil Geotérmico

```python
df = calculate_geotherm(...)

# Monotonía
assert (np.diff(df['T_K']) >= 0).all(), "T debe ser monótona creciente"
assert (np.diff(df['P_Pa']) >= 0).all(), "P debe ser monótona creciente"

# Rangos físicos
assert (df['T_K'] > 200).all() and (df['T_K'] < 5000).all()
assert (df['rho_kg_m3'] > 1000).all() and (df['rho_kg_m3'] < 10000).all()

# Comparar con Tierra si aplicable
if R_planet ≈ Re and M_total ≈ Me:
    T_moho = df[df['depth_km'] == 39]['T_K'].iloc[0]
    assert 800 < T_moho < 1100, f"T_moho = {T_moho} K fuera de rango"
```

---

## Cuando Pedir Ayuda al Usuario

1. **Modelos físicos nuevos**: Si necesitas implementar escalados diferentes a Hasterok o VRH
2. **Composiciones exóticas**: Minerales no incluidos en `MINERAL_PARAMS`
3. **Rangos P-T extremos**: BurnMan puede fallar fuera de ~0-200 GPa, 300-5000 K
4. **Interpretación geofísica**: Validar si resultados son físicamente razonables
5. **Parámetros de grid no documentados**: Config.py tiene parámetros de generación del grid

---

## Referencias Bibliográficas Clave

En carpeta `Textos/`:
- **Hasterok & Chapman (2011)**: Modelo de producción radiactiva exponencial
- **Turcotte & Schubert (2014)**: Geodinámica general, ecuaciones de conducción
- **Hofmeister (1999)**: Conductividad térmica de minerales
- Ver `Anteproyecto.pdf` para contexto completo del proyecto
