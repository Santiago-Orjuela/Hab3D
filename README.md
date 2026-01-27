# HAB3D  
## Seminario Trabajo de Grado  
## Evaluación de las condiciones de aguabilidad y habitabilidad en el interior de exoplanetas  

**Autor:** Santiago Andres Orjuela Montealegre

Repositorio del trabajo de grado enfocado en el estudio de la **habitabilidad y aguabilidad subsuperficial** de planetas tipo Tierra y super-Tierras, integrando modelos de estructura interna, evolución térmica y criterios físicos para la estabilidad de agua líquida.

---

## 📁 Estructura del repositorio

- `Codigos-Scripts/`  
  Scripts y notebooks principales para lectura del Planetary Grid, geotermia e índices de habitabilidad.
- `PlanetaryGrid/`  
  Grid de modelos planetarios organizados por **CMF/IMF**, con archivos `STRUC.dat` y `TEVOL.dat`.
- `Textos/`  
  Artículos, referencias y manuscritos asociados al anteproyecto y tesis.

---

## 📂 Scripts y notebooks

Este repositorio contiene scripts base para la lectura, procesamiento y análisis térmico de modelos planetarios generados a partir del *Planetary Grid*, así como notebooks de validación y demostración científica.

---

### `planetary_grid_reader.py`
**Descripción:**  
Módulo base para la lectura, procesamiento y extracción de propiedades físicas de modelos planetarios del *Planetary Grid*. Actúa como interfaz entre los archivos crudos del grid (`STRUC.dat`, `TEVOL.dat`) y los notebooks de análisis.

**Funcionalidades principales:**
- Lectura de archivos estructurales y térmicos:
  - `STRUC.dat`: estructura interna radial del planeta.
  - `TEVOL.dat`: evolución térmica temporal.
- Extracción automática de metadatos desde los encabezados:
  - Normalizaciones (`#norm={...}`).
  - Fracciones de capas (núcleo, manto, hielo).
- Cálculo de propiedades planetarias:
  - Perfil de gravedad \( g(r) \).
  - Masa y radio planetario.
  - Flujo de calor superficial.
- Procesamiento sistemático de modelos planetarios organizados por:
  - **CMF** (Core Mass Fraction).
  - **IMF** (Ice Mass Fraction).

**Funciones clave:**
- `read_struc_dat(path)`
- `read_tevol_dat(path)`
- `get_mass(path)`, `get_radius(path)`
- `get_surface_heat_flux(path_struc, path_tevol)`
- `process_planet_model(model_folder)`
- `process_all_models(planetary_grid_path, imf_filter=None)`

**Uso y validación:**  
Este script está diseñado para ser importado desde notebooks.  
Su correcto funcionamiento se **valida en `PlanetaryGrid.ipynb`**.

---

### `geotherm_calculator.py`
**Descripción:**  
Script para el cálculo de **geotermas planetarias 1D**, incorporando producción radiogénica, conductividad térmica efectiva y condiciones de frontera apropiadas para planetas rocosos.

**Funcionalidades principales:**
- Construcción de geotermas radiales a partir de:
  - Distribución interna de capas.
  - Producción de calor interna.
  - Propiedades térmicas del material.
- Cálculo de perfiles de temperatura y gradientes térmicos.
- Evaluación del estado térmico interno y su relación con:
  - Habitabilidad interna.
  - Estabilidad de capas y fases.

**Uso y validación:**  
No se ejecuta de forma independiente.  
Se importa y **valida dentro de `PlanetaryGrid.ipynb`**, junto con `planetary_grid_reader.py`.

---

### `habitability_calculator.py`
**Descripción:**  
Script base para el cálculo de **índices de aguabilidad y habitabilidad interna** en planetas rocosos, a partir de propiedades estructurales y térmicas del interior planetario.

Este módulo integra resultados provenientes de:
- Modelos estructurales del *Planetary Grid*.
- Geotermas planetarias.
- Propiedades físicas relevantes para la estabilidad de agua en el interior.

**Funcionalidades principales:**
- Cálculo de índices adimensionales de:
  - **Aguabilidad interna**.
  - **Habitabilidad interna**.
- Evaluación de regiones internas compatibles con:
  - Presión y temperatura adecuadas para agua líquida.
  - Persistencia temporal de condiciones favorables.
- Escalamiento de los índices en función de:
  - Masa planetaria.
  - Composición interna (CMF, IMF).
  - Estado térmico del planeta.

**Rol dentro del repositorio:**  
Constituye el **núcleo físico–conceptual** del trabajo, conectando los modelos internos con métricas cuantitativas de habitabilidad.

**Uso y validación:**  
Se importa y valida en notebooks de análisis, incluyendo:
- `Aguabilidad-Tierra.ipynb`
- Notebooks demostrativos adicionales.

---

### `PlanetaryGrid.ipynb`
**Descripción:**  
Notebook principal de análisis y validación de los modelos del *Planetary Grid*.

**Objetivos principales:**
- Validar la lectura correcta de los archivos del grid planetario.
- Verificar la consistencia de las propiedades físicas y térmicas calculadas.
- Explorar relaciones masa–radio, gravedad superficial y flujo de calor.
- Generar tablas de propiedades planetarias para análisis posteriores
  (aguabilidad, habitabilidad, escalamiento planetario, etc.).

**Scripts validados en este notebook:**
- `planetary_grid_reader.py`
- `geotherm_calculator.py`

---

### `Aguabilidad-Tierra.ipynb`
**Descripción:**  
Notebook de demostración y validación enfocado en el caso de la **Tierra** como referencia física.

**Objetivos principales:**
- Verificar el comportamiento del índice de aguabilidad interna.
- Calibrar y validar los criterios físicos del modelo.
- Comparar los resultados con expectativas geofísicas conocidas.

**Scripts utilizados y validados:**
- `habitability_calculator.py`
- Scripts base del *Planetary Grid*.

---

### `Geothermical_evolution.ipynb`
**Descripción:**  
Notebook de análisis de la **evolución térmica planetaria** a lo largo del tiempo geológico.

**Objetivos principales:**
- Analizar la evolución temporal del flujo de calor superficial.
- Estudiar el enfriamiento secular de planetas rocosos.
- Evaluar cómo la evolución térmica afecta la habitabilidad subsuperficial.
- Integrar datos de `TEVOL.dat` para reconstruir historias térmicas.

**Scripts utilizados:**
- `planetary_grid_reader.py` (lectura de datos de evolución térmica)
- `geotherm_calculator.py` (cálculo de geotermas en distintos tiempos)
- `habitability_calculator.py`

---

### `Indice_hab3D.ipynb`
**Descripción:**  
Notebook principal para el cálculo del **Índice de Habitabilidad 3D** (I₃D).

**Objetivos principales:**
- Definir e implementar el índice I₃D que cuantifica el volumen habitable subsuperficial.
- Integrar criterios de zona habitable circunestelar (Kopparapu et al. 2013).
- Evaluar cómo el índice varía con:
  - Masa planetaria
  - Flujo de calor interno
  - Distancia orbital
- Generar mapas de habitabilidad 3D para diferentes configuraciones planetarias.

**Scripts utilizados:**
- `habitability_calculator.py` (funciones de zona habitable y aguabilidad)
- `geotherm_calculator.py` (perfiles térmicos)
- `planetary_grid_reader.py`

---

### `Trappist-1.ipynb`
**Descripción:**  
Caso de estudio aplicado al exoplaneta **TRAPPIST-1e**, un planeta rocoso potencialmente habitable en la zona habitable de su estrella.

**Objetivos principales:**
- Aplicar los modelos de habitabilidad subsuperficial a un caso real.
- Calcular perfiles geotérmicos con parámetros específicos de TRAPPIST-1e:
  - Masa: 0.692 M⊕
  - Radio: 0.92 R⊕
  - Distancia orbital: 0.02925 AU
  - Estrella: enana M (L = 0.000566 L☉)
- Evaluar zonas de aguabilidad y habitabilidad interna.
- Comparar con el caso terrestre.
- Implementar modelos de estructura interna usando BurnMan.

**Scripts utilizados:**
- `habitability_calculator.py`
- `geotherm_calculator.py`
- Composiciones minerales ajustadas para super-Tierras

---

## ⚙️ Requisitos

Python 3.10+ y los paquetes:

- `numpy`, `pandas`, `matplotlib`, `astropy`
- `burnman` (mineralogía / ecuaciones de estado)

Instalación rápida (entorno local):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
