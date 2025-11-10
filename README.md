# Hab3D

Repositorio del trabajo de grado sobre habitabilidad y "aguabilidad" subsuperficial de planetas tipo Tierra.

## Estructura

- `Codigos-Scripts/`: scripts y notebooks principales (PlanetaryGrid, geotermia, HZ, TRAPPIST-1e).
- `PlanetaryGrid/`: grid de modelos CMF/IMF y utilidades.
- `Textos/`: artículos, referencias y manuscritos (anteproyecto/tesis).

## Componentes clave

- `Codigos-Scripts/planetary_grid_reader.py`: lectura y procesamiento del grid (STRUC/TEVOL), g(r), flujo de calor superficial.
- `Codigos-Scripts/geotherm_calculator.py`: perfil geotérmico conductivo 1D con calentamiento radiogénico y conductividad VRH acoplada a BurnMan.
- `Codigos-Scripts/Aguabilidad-Tierra.ipynb`: análisis y figuras (ZH de Kopparapu 2013, validaciones terrestres).

## Requisitos

Python 3.10+ y los paquetes:

- numpy, pandas, matplotlib, astropy
- burnman (mineralogía / EOS)

Instalación rápida (entorno local):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

## Objetivo

Desarrollar y evaluar criterios físicos para la existencia de una Capa de Aguabilidad (agua líquida estable) en la subsuperficie de exoplanetas rocosos, integrando estructura interna (CMF/IMF), evolución térmica, y conducción geotérmica.

## Licencia

MIT (por definir según necesidades del proyecto).
