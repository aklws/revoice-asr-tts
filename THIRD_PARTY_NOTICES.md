# Third-Party Notices

This repository contains or depends on third-party software, models, binaries, and other assets that are not covered by the root [LICENSE](./LICENSE) unless explicitly stated otherwise.

## Scope

The root MIT license applies only to the original code and documentation authored for this repository.

The following categories are excluded from the root MIT license when present in this repository or distributed alongside it:

- Vendored third-party source code
- Bundled third-party binaries
- Model weights, checkpoints, and related assets
- Files or directories that include their own license, notice, or upstream terms

## Known Third-Party Components

### `vendor/index-tts/`

- Source: `https://github.com/index-tts/index-tts`
- Local license files:
  - `vendor/index-tts/LICENSE`
  - `vendor/index-tts/LICENSE_ZH.txt`
- License summary:
  - This component is not distributed under the root MIT license.
  - The local license file identifies it as the `bilibili Model Use License Agreement`.
  - Any use, redistribution, modification, or downstream distribution of this component must comply with its upstream terms.

### `bin/ffmpeg/`

- This directory contains FFmpeg binaries and related files.
- It is not covered by the root MIT license.
- Please review the bundled license files in that directory before redistribution.

### `bin/Qwen3-ASR-Transcribe/`

- This directory contains third-party runtime files, binaries, and related dependencies for the ASR stack.
- It is not covered by the root MIT license unless a specific file states otherwise.
- Before redistribution, review the upstream project terms and any bundled license files within that directory.

### `checkpoints/`

- Model files, checkpoints, tokenizer files, and related assets are not covered by the root MIT license.
- Their use and redistribution depend on the corresponding upstream model or dataset terms.

## Your Responsibility

If you redistribute this repository, a derived package, or any bundled runtime assets, you are responsible for:

- preserving original copyright and license notices
- complying with upstream license terms
- checking whether model weights, binaries, and vendored code allow redistribution

## Practical Rule

If a file or directory includes its own license, notice, or upstream attribution, treat that file or directory as governed by its own terms rather than the root MIT license.
