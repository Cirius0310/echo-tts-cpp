<# 
.EXPORT
    package_portable.ps1 — Create a self-contained portable release of echo-tts

.DESCRIPTION
    Assembles the build output, CUDA runtime DLLs, cuDNN DLLs, OpenSSL,
    VC++ redistributables, and optionally ffmpeg into a single ZIP archive
    that can be extracted and run without installing any dependencies.

.PARAMETER BuildDir
    Path to the CMake build output directory (default: cpp/build/Release)

.PARAMETER OutputZip
    Path for the output ZIP (default: echo-tts-portable-<timestamp>.zip)

.PARAMETER CudaVersion
    CUDA major version to bundle (12 or 13, auto-detected if not specified)

.PARAMETER WithFfmpeg
    Download and include ffmpeg portable binaries (default: $true)

.PARAMETER OpenSslDir
    Override path to OpenSSL bin directory (auto-detected on Windows)

.EXAMPLE
    .\scripts\package_portable.ps1
    .\scripts\package_portable.ps1 -CudaVersion 12 -WithFfmpeg:$false
#>

param(
    [string]$BuildDir,
    [string]$OutputZip,
    [int]$CudaVersion = 0,
    [bool]$WithFfmpeg = $true
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot

# Default paths relative to project root
if (-not $BuildDir) { $BuildDir = Join-Path $ProjectRoot "cpp\build\Release" }
if (-not $OutputZip) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputZip = Join-Path $ProjectRoot "echo-tts-portable-$timestamp.zip"
}

Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Echo-TTS Portable Release Packager" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# -- Validate build directory --
$ExePath = Join-Path $BuildDir "echo-tts.exe"
if (-not (Test-Path $ExePath)) {
    Write-Error "echo-tts.exe not found in $BuildDir — build first with: cmake --build cpp/build --config Release"
    exit 1
}
Write-Host "[OK] Build directory: $BuildDir" -ForegroundColor Green

# -- Create staging directory --
$StagingDir = Join-Path $env:TEMP "echo-tts-portable-$(Get-Random)"
New-Item -ItemType Directory -Force -Path $StagingDir | Out-Null
Write-Host "[OK] Staging: $StagingDir" -ForegroundColor Green

# -- Step 1: Copy build output --
Write-Host ""
Write-Host "Step 1: Copying build output..." -ForegroundColor Yellow
$copiedBuild = 0
Get-ChildItem "$BuildDir\*" -Include "*.exe", "*.dll" | ForEach-Object {
    Copy-Item $_.FullName $StagingDir
    Write-Host "  + $($_.Name)"
    $copiedBuild++
}
Write-Host "  → Copied $copiedBuild files" -ForegroundColor Green

# -- Step 2: Auto-discover CUDA toolkit --
Write-Host ""
Write-Host "Step 2: Locating CUDA toolkit DLLs..." -ForegroundColor Yellow

function Find-CudaBin {
    param([int]$Major)
    if ($env:CUDA_PATH) {
        $cand = $env:CUDA_PATH -replace '\\$'
        if (Test-Path (Join-Path $cand "bin\x64\cublas64_$Major.dll")) { return Join-Path $cand "bin\x64" }
        if (Test-Path (Join-Path $cand "bin\cublas64_$Major.dll")) { return Join-Path $cand "bin" }
    }
    $base = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $base) {
        $dirs = Get-ChildItem $base -Directory | Sort-Object Name -Descending
        foreach ($d in $dirs) {
            foreach ($sub in @("bin\x64", "bin")) {
                $bindir = Join-Path $d.FullName $sub
                if (Test-Path (Join-Path $bindir "cublas64_$Major.dll")) { return $bindir }
            }
        }
    }
    return $null
}

# Detect which CUDA major versions are required by the bundled libraries
$requiredCudaVersions = @()
if ($CudaVersion -gt 0) {
    $requiredCudaVersions += $CudaVersion
} else {
    # Auto-detect from ONNX Runtime provider DLL (present in build dir after cmake copies it)
    $onnxProviderPath = Join-Path $BuildDir "onnxruntime_providers_cuda.dll"
    if (Test-Path $onnxProviderPath) {
        $bytes = [System.IO.File]::ReadAllBytes($onnxProviderPath)
        $text = [System.Text.Encoding]::ASCII.GetString($bytes)
        for ($v = 14; $v -ge 10; $v--) {
            if ($text -match "cublas64_$v\.") {
                Write-Host "  ONNX Runtime needs CUDA $v.x" -ForegroundColor Gray
                $requiredCudaVersions += $v
                break
            }
        }
    }
    # Also detect from ggml-cuda.dll
    $ggmlCudaPath = Join-Path $BuildDir "ggml-cuda.dll"
    if (Test-Path $ggmlCudaPath) {
        $bytes = [System.IO.File]::ReadAllBytes($ggmlCudaPath)
        $text = [System.Text.Encoding]::ASCII.GetString($bytes)
        for ($v = 14; $v -ge 10; $v--) {
            if ($text -match "cublas64_$v\.") {
                Write-Host "  ggml-cuda needs CUDA $v.x" -ForegroundColor Gray
                if ($requiredCudaVersions -notcontains $v) { $requiredCudaVersions += $v }
                break
            }
        }
    }
    # Fallback
    if ($requiredCudaVersions.Count -eq 0) { $requiredCudaVersions += 12 }
}

$requiredCudaVersions = $requiredCudaVersions | Sort-Object -Unique
Write-Host "  Required CUDA version(s): $($requiredCudaVersions -join ', ')" -ForegroundColor Green

$cudaCopied = 0
foreach ($cv in $requiredCudaVersions) {
    $CudaBin = Find-CudaBin -Major $cv
    if (-not $CudaBin) {
        Write-Error "CUDA $cv.x toolkit not found. Install it or set with -CudaVersion"
        exit 1
    }
    Write-Host "  CUDA $cv.x from: $CudaBin" -ForegroundColor Green

    $cudaDlls = @("cudart64_$cv.dll", "cublas64_$cv.dll", "cublasLt64_$cv.dll")
    foreach ($dll in $cudaDlls) {
        $src = Join-Path $CudaBin $dll
        if (Test-Path $src) {
            if (-not (Test-Path (Join-Path $StagingDir $dll))) {
                Copy-Item $src $StagingDir
                Write-Host "  + $dll"
                $cudaCopied++
            }
        } else {
            Write-Warning "  Could not find: $dll"
        }
    }

    $cufftPat = "cufft64_*.dll"
    $cufftDll = Get-ChildItem $CudaBin -Filter $cufftPat | Select-Object -First 1
    if ($cufftDll -and -not (Test-Path (Join-Path $StagingDir $cufftDll.Name))) {
        Copy-Item $cufftDll.FullName $StagingDir
        Write-Host "  + $($cufftDll.Name)"
        $cudaCopied++
    }
}
Write-Host "  → Copied $cudaCopied CUDA DLLs" -ForegroundColor Green

# -- Step 3: Auto-discover cuDNN --
Write-Host ""
Write-Host "Step 3: Locating cuDNN DLLs..." -ForegroundColor Yellow

function Find-CudnnBin {
    param([int]$CudaMajor)
    $base = "C:\Program Files\NVIDIA\CUDNN"
    if (Test-Path $base) {
        $verdirs = Get-ChildItem $base -Directory | Sort-Object Name -Descending
        foreach ($vd in $verdirs) {
            $bin = Join-Path $vd.FullName "bin"
            if (Test-Path $bin) {
                # Look for subdirs with x64 DLLs, prefer matching CUDA major version
                $subs = Get-ChildItem $bin -Directory | Sort-Object Name -Descending
                foreach ($sub in $subs) {
                    if ($CudaMajor -gt 0 -and $sub.Name -like "$CudaMajor.*") {
                        # Prefer exact CUDA match first
                        $x64 = Join-Path $sub.FullName "x64"
                        if (Test-Path (Join-Path $x64 "cudnn64_9.dll")) {
                            return $x64
                        }
                    }
                }
                # Fallback: any subdir with x64 DLLs
                foreach ($sub in $subs) {
                    $x64 = Join-Path $sub.FullName "x64"
                    if (Test-Path (Join-Path $x64 "cudnn64_9.dll")) {
                        return $x64
                    }
                }
            }
        }
    }
    return $null
}

$CudnnBin = $null
foreach ($cv in $requiredCudaVersions) {
    $CudnnBin = Find-CudnnBin -CudaMajor $cv
    if ($CudnnBin) { break }
}
if (-not $CudnnBin -and $requiredCudaVersions.Count -gt 0) {
    $CudnnBin = Find-CudnnBin -CudaMajor 0  # any version
}
if ($CudnnBin) {
    Write-Host "  Found: $CudnnBin" -ForegroundColor Green
    $cudnnCopied = 0
    Get-ChildItem $CudnnBin -Filter "*.dll" | ForEach-Object {
        Copy-Item $_.FullName $StagingDir
        Write-Host "  + $($_.Name)"
        $cudnnCopied++
    }
    Write-Host "  → Copied $cudnnCopied cuDNN DLLs" -ForegroundColor Green
} else {
    Write-Warning "  cuDNN not found. ONNX Runtime CUDA provider will fail without it."
    Write-Warning "  Download from: https://developer.nvidia.com/cudnn"
}

# -- Step 4: Locate OpenSSL --
Write-Host ""
Write-Host "Step 4: Locating OpenSSL DLLs..." -ForegroundColor Yellow

function Find-OpenSslBin {
    $paths = @(
        "C:\Program Files\OpenSSL-Win64\bin",
        "C:\Program Files\OpenSSL\bin",
        "C:\OpenSSL-Win64\bin"
    )
    foreach ($p in $paths) {
        if (Test-Path (Join-Path $p "libssl-3-x64.dll")) { return $p }
        if (Test-Path (Join-Path $p "libssl-1_1-x64.dll")) { return $p }
    }
    # Search PATH
    $pathDirs = $env:PATH -split ';'
    foreach ($d in $pathDirs) {
        if (Test-Path (Join-Path $d "libssl-3-x64.dll")) { return $d }
    }
    return $null
}

$OpenSslBin = Find-OpenSslBin
if ($OpenSslBin) {
    Write-Host "  Found: $OpenSslBin" -ForegroundColor Green
    $sslDlls = @("libssl-3-x64.dll", "libcrypto-3-x64.dll", "libssl-1_1-x64.dll", "libcrypto-1_1-x64.dll")
    $sslCopied = 0
    foreach ($dll in $sslDlls) {
        $src = Join-Path $OpenSslBin $dll
        if (Test-Path $src) {
            Copy-Item $src $StagingDir
            Write-Host "  + $dll"
            $sslCopied++
        }
    }
    Write-Host "  → Copied $sslCopied OpenSSL DLLs" -ForegroundColor Green
} else {
    Write-Warning "  OpenSSL not found. HTTPS in server mode will fail."
    Write-Warning "  Install from: https://slproweb.com/products/Win32OpenSSL.html"
}

# -- Step 5: Copy VC++ redistributables --
Write-Host ""
Write-Host "Step 5: Copying VC++ redistributables..." -ForegroundColor Yellow

$system32 = [Environment]::GetFolderPath('System')
$vcredistDlls = @("MSVCP140.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll", "VCOMP140.dll")
$vcrCopied = 0
foreach ($dll in $vcredistDlls) {
    $src = Join-Path $system32 $dll
    if (Test-Path $src) {
        Copy-Item $src $StagingDir
        Write-Host "  + $dll"
        $vcrCopied++
    } else {
        Write-Warning "  $dll not found in $system32 — may need VC++ 2019 redist installed"
    }
}
Write-Host "  → Copied $vcrCopied VC++ DLLs" -ForegroundColor Green

# -- Step 6: Download ffmpeg (optional) --
if ($WithFfmpeg) {
    Write-Host ""
    Write-Host "Step 6: Fetching ffmpeg portable..." -ForegroundColor Yellow
    try {
        $ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        $ffmpegZip = Join-Path $env:TEMP "ffmpeg-portable.zip"
        $ffmpegDir = Join-Path $env:TEMP "ffmpeg-portable-extract"

        Write-Host "  Downloading ffmpeg..." -ForegroundColor Gray
        Invoke-WebRequest -Uri $ffmpegUrl -OutFile $ffmpegZip -UseBasicParsing

        Write-Host "  Extracting..." -ForegroundColor Gray
        Expand-Archive -Path $ffmpegZip -DestinationPath $ffmpegDir -Force

        # Find ffmpeg.exe (it's nested in a versioned subfolder)
        $ffmpegExe = Get-ChildItem $ffmpegDir -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
        $ffprobeExe = Get-ChildItem $ffmpegDir -Recurse -Filter "ffprobe.exe" | Select-Object -First 1

        if ($ffmpegExe) {
            Copy-Item $ffmpegExe.FullName $StagingDir
            Write-Host "  + ffmpeg.exe" -ForegroundColor Green
        }
        if ($ffprobeExe) {
            Copy-Item $ffprobeExe.FullName $StagingDir
            Write-Host "  + ffprobe.exe" -ForegroundColor Green
        }

        Remove-Item $ffmpegZip -Force -ErrorAction SilentlyContinue
        Remove-Item $ffmpegDir -Recurse -Force -ErrorAction SilentlyContinue
    } catch {
        Write-Warning "  ffmpeg download failed: $_"
        Write-Warning "  MP3 output will be unavailable. Install ffmpeg separately."
    }
}

# -- Step 7: Write portable README --
Write-Host ""
Write-Host "Step 7: Creating portable README..." -ForegroundColor Yellow

$readmeContent = @'
Echo-TTS Portable Release
=========================

This is a self-contained portable release of Echo-TTS.
All GPU runtime libraries (CUDA, cuDNN, etc.) are included.

Requirements:
  - NVIDIA GPU with driver 550+ (supports CUDA 12.x)
  - Model files (downloaded separately - see below)

Quick Start:
  1. Download model files from HuggingFace:
     https://huggingface.co/tmdarkbr/echo-tts-gguf
     - Place echo-dit.gguf in this folder
     - Place onnx_models/ folder next to echo-tts.exe

  2. Run the server:
     echo-tts serve --model echo-dit.gguf --voice NAME=PATH --dac-encoder onnx_models/dac_encoder.onnx --dac-decoder onnx_models/dac_decoder.onnx

  3. Or generate a single file:
     echo-tts --model echo-dit.gguf --speaker myvoice.wav --text "Hello!" --output out.wav --dac-encoder onnx_models/dac_encoder.onnx --dac-decoder onnx_models/dac_decoder.onnx

No installation needed - all libraries run from this folder.

For full documentation, visit: https://github.com/Cirius0310/echo-tts-cpp
'@

$readmePath = Join-Path $StagingDir "README.txt"
Set-Content -Path $readmePath -Value $readmeContent
Write-Host "  + README.txt" -ForegroundColor Green

# -- Step 8: Create ZIP archive --
Write-Host ""
Write-Host "Step 8: Creating ZIP archive..." -ForegroundColor Yellow

if (Test-Path $OutputZip) { Remove-Item $OutputZip -Force }

$stagingFiles = Get-ChildItem $StagingDir
Write-Host "  Packing $($stagingFiles.Count) files..." -ForegroundColor Gray

Compress-Archive -Path "$StagingDir\*" -DestinationPath $OutputZip -Force

# -- Summary before cleanup --
$zipSize = (Get-Item $OutputZip).Length
$stagingFiles = Get-ChildItem $StagingDir
Write-Host ""
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Portable release created!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Path:  $OutputZip" -ForegroundColor White
Write-Host "  Size:  $([math]::Round($zipSize / 1MB, 1)) MB" -ForegroundColor White
Write-Host "  Files: $($stagingFiles.Count)" -ForegroundColor White
Write-Host "  CUDA:  v$CudaVersion.x" -ForegroundColor White
Write-Host ""
Write-Host "Contents:"
$stagingFiles | ForEach-Object { Write-Host "  $($_.Name)" }

# -- Cleanup --
Remove-Item $StagingDir -Recurse -Force -ErrorAction SilentlyContinue
