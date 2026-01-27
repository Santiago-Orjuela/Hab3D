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
Script para el cálculo de **geotermas planetarias 1D**, incorporando producción radiogénica, conductividad térmica efectiva y condiciones de frontera apropiadas para planetas rocosos y ricos en hielo.

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
Script base para el cálculo de **índices de aguabilidad y habitabilidad interna** en planetas rocosos y ricos en hielo, a partir de propiedades estructurales y térmicas del interior planetario.

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

### Notebooks demostrativos adicionales
**Descripción:**  
Notebooks exploratorios donde se aplica `habitability_calculator.py` a distintos escenarios planetarios (variaciones en masa, composición y estado térmico).  
Se documentan individualmente a medida que se incorporan al repositorio.

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
