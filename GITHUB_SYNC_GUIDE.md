# 🚀 Guía Rápida de Sincronización con GitHub

## Archivos Excluidos del Repositorio

### ❌ Archivos/carpetas que NO se suben a GitHub:

**Documentos de trabajo:**
- `Tareas.md`
- `Temas de Discusión.md`

**Carpetas LaTeX:**
- `Textos/Proyecto/`
- `Textos/Anteproyecto/`

**Notebooks exploratorios:**
- `Codigos-Scripts/BurnMan.ipynb`
- `Codigos-Scripts/phase_diagram_salinity.ipynb`
- `Codigos-Scripts/TRAPPIST-1e.ipynb`

**Archivos temporales/cache:**
- `__pycache__/`
- `.ipynb_checkpoints/`
- `*.pyc`, `*.pyo`
- `*.dat`, `*.csv`, `*.npz`, `*.pkl`
- Archivos de build de LaTeX

---

## ✅ Archivos Incluidos en el Repositorio

### Scripts principales:
- `planetary_grid_reader.py`
- `geotherm_calculator.py`
- `habitability_calculator.py`

### Notebooks principales:
- `PlanetaryGrid.ipynb` - Validación del grid planetario
- `Aguabilidad-Tierra.ipynb` - Caso de estudio: Tierra
- `Geothermical_evolution.ipynb` - Evolución térmica
- `Indice_hab3D.ipynb` - Índice de habitabilidad 3D
- `Trappist-1.ipynb` - Caso de estudio: TRAPPIST-1e

### Documentación:
- `README.md`
- `.gitignore`
- `requirements.txt`
- Esta guía

---

## 🔧 Métodos de Sincronización

### Método 1: Script Automatizado (Recomendado)

Ejecuta el script de PowerShell:

```powershell
cd "c:\Users\Usuario Cs\OneDrive\Documents\Universidad\Hab3d"
.\sync_github.ps1
```

El script hace automáticamente:
1. ✓ Verifica el estado del repositorio
2. ✓ Elimina archivos no deseados del tracking
3. ✓ Agrega archivos actualizados
4. ✓ Muestra el estado final
5. ✓ Opcionalmente hace commit y push

---

### Método 2: Comandos Manuales

Si prefieres hacerlo manualmente:

#### 1. Eliminar archivos no deseados del repositorio:

```powershell
# Eliminar archivos individuales
git rm --cached Tareas.md
git rm --cached "Temas de Discusión.md"
git rm --cached Codigos-Scripts/BurnMan.ipynb
git rm --cached Codigos-Scripts/phase_diagram_salinity.ipynb
git rm --cached Codigos-Scripts/TRAPPIST-1e.ipynb

# Eliminar carpetas completas
git rm -r --cached Textos/Proyecto/
git rm -r --cached Textos/Anteproyecto/
```

#### 2. Agregar archivos actualizados:

```powershell
git add README.md
git add .gitignore
git add requirements.txt
git add Codigos-Scripts/*.py
git add Codigos-Scripts/PlanetaryGrid.ipynb
git add Codigos-Scripts/Aguabilidad-Tierra.ipynb
git add Codigos-Scripts/Geothermical_evolution.ipynb
git add Codigos-Scripts/Indice_hab3D.ipynb
git add Codigos-Scripts/Trappist-1.ipynb
git add Codigos-Scripts/TRAPPIST-1e.ipynb
```

#### 3. Hacer commit y push:

```powershell
git commit -m "Actualizar repositorio - reorganización de archivos"
git push origin main
```

---

## 📝 Verificaciones Importantes

### Antes de hacer push, verifica:

```powershell
# Ver estado del repositorio
git status

# Ver archivos que se subirán
git diff --cached --name-only

# Ver archivos rastreados
git ls-files

# Ver archivos ignorados
git status --ignored
```

### Si un archivo ignorado sigue apareciendo:

```powershell
# Eliminar del cache de Git
git rm --cached nombre_del_archivo

# Hacer commit del cambio
git commit -m "Eliminar archivo del repositorio"
```

---

## 🔄 Workflow Recomendado

### Para trabajo diario:

1. **Antes de empezar a trabajar:**
   ```powershell
   git pull origin main
   ```

2. **Durante el trabajo:**
   - Trabaja normalmente
   - Los archivos en `.gitignore` no se rastrearán automáticamente

3. **Al finalizar el día:**
   ```powershell
   # Opción A: Usar el script
   .\sync_github.ps1

   # Opción B: Comandos manuales
   git add <archivos_modificados>
   git commit -m "Descripción de cambios"
   git push origin main
   ```

---

## ⚠️ Notas Importantes

1. **El `.gitignore` ya está configurado** para excluir automáticamente:
   - Archivos de cache (`__pycache__`, `.pyc`)
   - Checkpoints de Jupyter (`.ipynb_checkpoints/`)
   - Archivos de datos grandes (`*.dat`, `*.csv`, `*.pkl`)
   - Build artifacts de LaTeX

2. **Los archivos de datos del PlanetaryGrid:**
   - Los archivos `.dat` están excluidos por defecto
   - Solo se sube la estructura de carpetas (`Config.py`)
   - Esto mantiene el repositorio liviano

3. **Si necesitas agregar un archivo que está en `.gitignore`:**
   ```powershell
   git add -f nombre_del_archivo
   ```

---

## 🆘 Solución de Problemas

### Problema: "El archivo sigue apareciendo aunque lo agregué a .gitignore"

**Solución:**
```powershell
git rm --cached nombre_del_archivo
git commit -m "Eliminar archivo del tracking"
git push origin main
```

### Problema: "Muchos archivos sin rastrear aparecen en git status"

**Solución:** Verifica que `.gitignore` esté actualizado y haz:
```powershell
git status --ignored  # Ver qué archivos están siendo ignorados
```

### Problema: "Error al hacer push"

**Soluciones comunes:**
```powershell
# 1. Hacer pull primero para sincronizar
git pull origin main

# 2. Si hay conflictos, resolverlos y hacer commit
git add .
git commit -m "Resolver conflictos"

# 3. Intentar push nuevamente
git push origin main
```

---

## 📞 Recursos Adicionales

- **GitHub del proyecto:** https://github.com/tu-usuario/Hab3d
- **Documentación de Git:** https://git-scm.com/doc
- **README del proyecto:** [README.md](README.md)

---

**Última actualización:** Enero 2026  
**Autor:** Santiago Orjuela
