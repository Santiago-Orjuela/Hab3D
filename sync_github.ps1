# ============================================================================
# Script de sincronización con GitHub para HAB3D
# ============================================================================
# Este script ayuda a mantener sincronizado el repositorio local con GitHub,
# eliminando archivos no deseados y actualizando el repositorio remoto.
#
# Autor: Santiago Orjuela
# Fecha: Enero 2026
# ============================================================================

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  HAB3D - Sincronización con GitHub" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar que estamos en el directorio correcto
$currentDir = Get-Location
$expectedPath = "Hab3d"
if (-not ($currentDir.Path -match $expectedPath)) {
    Write-Host "⚠️  ADVERTENCIA: No estás en el directorio Hab3d" -ForegroundColor Yellow
    Write-Host "Directorio actual: $currentDir" -ForegroundColor Yellow
    $continue = Read-Host "¿Deseas continuar de todas formas? (s/n)"
    if ($continue -ne "s") {
        exit
    }
}

Write-Host "📂 Directorio de trabajo: $currentDir" -ForegroundColor Green
Write-Host ""

# ============================================================================
# PASO 1: Verificar estado de Git
# ============================================================================
Write-Host "🔍 PASO 1: Verificando estado del repositorio..." -ForegroundColor Yellow
Write-Host ""

git status

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# PASO 2: Eliminar archivos no deseados del repositorio (si existen)
# ============================================================================
Write-Host "🗑️  PASO 2: Eliminando archivos no deseados del repositorio..." -ForegroundColor Yellow
Write-Host ""

$filesToRemove = @(
    "Tareas.md",
    "Temas de Discusión.md",
    "Codigos-Scripts/BurnMan.ipynb",
    "Codigos-Scripts/phase_diagram_salinity.ipynb",
    "Codigos-Scripts/TRAPPIST-1e.ipynb",
    "Textos/Proyecto/",
    "Textos/Anteproyecto/"
)

$removedFiles = @()
foreach ($file in $filesToRemove) {
    # Verificar si el archivo está rastreado por Git
    $gitCheck = git ls-files $file 2>$null
    if ($gitCheck) {
        Write-Host "  ➜ Eliminando de Git: $file" -ForegroundColor Cyan
        git rm -r --cached $file 2>$null
        if ($?) {
            $removedFiles += $file
        }
    } else {
        Write-Host "  ✓ Archivo ya no rastreado: $file" -ForegroundColor Gray
    }
}

if ($removedFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "✅ Archivos eliminados del repositorio:" -ForegroundColor Green
    foreach ($file in $removedFiles) {
        Write-Host "   - $file" -ForegroundColor Green
    }
} else {
    Write-Host ""
    Write-Host "✅ No hay archivos para eliminar del repositorio" -ForegroundColor Green
}

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# PASO 3: Agregar archivos actualizados
# ============================================================================
Write-Host "📝 PASO 3: Agregando archivos actualizados..." -ForegroundColor Yellow
Write-Host ""

# Archivos específicos a agregar
$filesToAdd = @(
    "README.md",
    ".gitignore",
    "requirements.txt",
    "Codigos-Scripts/planetary_grid_reader.py",
    "Codigos-Scripts/geotherm_calculator.py",
    "Codigos-Scripts/habitability_calculator.py",
    "Codigos-Scripts/PlanetaryGrid.ipynb",
    "Codigos-Scripts/Aguabilidad-Tierra.ipynb",
    "Codigos-Scripts/Geothermical_evolution.ipynb",
    "Codigos-Scripts/Indice_hab3D.ipynb",
    "Codigos-Scripts/Trappist-1.ipynb"
)

foreach ($file in $filesToAdd) {
    if (Test-Path $file) {
        git add $file
        Write-Host "  ✓ Agregado: $file" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  No encontrado: $file" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# PASO 4: Mostrar estado actualizado
# ============================================================================
Write-Host "📊 PASO 4: Estado actualizado del repositorio..." -ForegroundColor Yellow
Write-Host ""

git status

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# PASO 5: Confirmar cambios
# ============================================================================
Write-Host "💾 PASO 5: ¿Deseas hacer commit de estos cambios?" -ForegroundColor Yellow
Write-Host ""
Write-Host "Opciones:" -ForegroundColor White
Write-Host "  1) Hacer commit y push" -ForegroundColor White
Write-Host "  2) Solo hacer commit (sin push)" -ForegroundColor White
Write-Host "  3) Cancelar (no hacer commit)" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Selecciona una opción (1/2/3)"

if ($choice -eq "1" -or $choice -eq "2") {
    Write-Host ""
    $commitMessage = Read-Host "Mensaje del commit"
    
    if ([string]::IsNullOrWhiteSpace($commitMessage)) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
        $commitMessage = "Actualizacion del repositorio - $timestamp"
    }
    
    Write-Host ""
    Write-Host "📦 Haciendo commit..." -ForegroundColor Yellow
    git commit -m "$commitMessage"
    
    if ($?) {
        Write-Host "✅ Commit realizado exitosamente" -ForegroundColor Green
        
        if ($choice -eq "1") {
            Write-Host ""
            Write-Host "🚀 Subiendo cambios a GitHub..." -ForegroundColor Yellow
            git push origin main
            
            if ($?) {
                Write-Host "✅ Cambios subidos a GitHub exitosamente" -ForegroundColor Green
            } else {
                Write-Host "❌ Error al subir cambios a GitHub" -ForegroundColor Red
                Write-Host "Intenta manualmente: git push origin main" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "❌ Error al hacer commit" -ForegroundColor Red
    }
} else {
    Write-Host ""
    Write-Host "⏸️  Operación cancelada. Los archivos están staged pero no se hizo commit." -ForegroundColor Yellow
    Write-Host "Para hacer commit manualmente, usa: git commit -m `"tu mensaje`"" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  Sincronización completada" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Comandos útiles adicionales:" -ForegroundColor White
Write-Host "  - Ver estado: git status" -ForegroundColor Gray
Write-Host "  - Ver historial: git log --oneline" -ForegroundColor Gray
Write-Host "  - Ver archivos rastreados: git ls-files" -ForegroundColor Gray
Write-Host ""
