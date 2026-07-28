# Revoice ASR-TTS Packaging

## Build Target

- Platform: `Windows`
- Packaging mode: `Cython` bundle + launcher script
- Entry point: `main.py`

## Build Flow

The current Windows packaging flow no longer uses `PyInstaller` or `Nuitka`.

1. [build.ps1](../packaging/windows/build.ps1) compiles local source packages such as `app` and `ui` into `.pyd` modules and assembles the runtime payload under `dist/`
2. [build-installer.ps1](../packaging/windows/build-installer.ps1) builds the installer from that payload

Run from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\windows\build.ps1
powershell -ExecutionPolicy Bypass -File .\packaging\windows\build-installer.ps1
```

## Output Layout

After a successful build, the package directory looks like this:

```text
dist/
  Revoice-ASR-TTS-v<version>/
    Launch-Revoice-ASR-TTS.bat
    main.py
    .env.example
    README.md
    README.en.md
    app/
    ui/
    vendor/
    asset/
    bin/
      ffmpeg/
    checkpoints/
    logs/
    outputs/
```

## Runtime Behavior

- The launcher script runs `uv sync --frozen` before starting `main.py`
- The packaged app resolves `.env`, `checkpoints`, `bin`, `asset`, `logs`, and `outputs` relative to the package directory
- `UPX` post-pack compression is optional and only applied to selected low-risk files
- The packaging script does not bundle the local `bin/ffmpeg/` directory by default; use `-IncludeBundledFfmpeg` only when you intentionally want to redistribute that third-party runtime

## What Is Bundled

- Cython-compiled local application modules
- Vendored runtime sources that remain as Python files where needed
- `README.md`, `README.en.md`, and `.env.example`
- Non-audio assets under `asset/`

## What Is Not Bundled

- Local model directories under `checkpoints/`
- Local runtime binaries under `bin/` unless you explicitly choose to redistribute them
- Sample/reference audio files under `asset/`
- Any later downloaded or replaced model weights

## Deployment Notes

1. Build the package
2. Optionally create a `.env` next to the launcher based on `.env.example`
3. Copy required model directories into `dist/Revoice-ASR-TTS-v<version>/checkpoints/`, or let the app prepare them on first launch
4. Prepare your own reference audio files locally when using the packaged app

## Notes

- `vendor/index-tts` is intentionally kept as source form for compatibility with reflection and runtime code paths
- The packaged build is designed around `uv`, so the target machine needs `uv` available in `PATH`

- If you want a single-file build later, expect slower startup and more path-handling edge cases. `standalone/onedir` remains the safer release format for this project.
