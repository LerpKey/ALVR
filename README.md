# ALVR Cloud — Dynamic Foveated Rendering Research Fork

This repository is an experimental research fork of [ALVR](https://github.com/alvr-org/ALVR), focused primarily on eye-tracked dynamic foveated rendering (DFR).

It is not a current upstream ALVR release and should not be expected to be compatible with the latest upstream code or headset runtimes.

## Fork provenance

The code lineage starts from the official ALVR `master` branch at commit [`1f0ba243`](https://github.com/alvr-org/ALVR/commit/1f0ba243c0fd46cb0b54e2e54d46719a45fc1664), dated 2025-05-09:

> Remove PrefersNonDefaultGPU from desktop files (#2812)

The first local development snapshot was created directly on top of that upstream commit. This repository therefore represents an older ALVR baseline, not a fork of the current upstream tip.

The public history intentionally contains only two source-focused commits: the DFR implementation and the later telemetry/data-collection additions. Intermediate experiment commits and private research materials are not included.

## Main contribution

The primary contribution is an experimental per-eye, eye-tracked DFR pipeline:

- PICO eye-tracking data acquisition and transport through the Android OpenXR client.
- Per-eye gaze shifts for server-side foveated encoding.
- Client-side inverse foveation to reconstruct the streamed image.
- Explicit `Unsupported`, `Standby`, and `Active` eye-tracking states.
- Fixed foveated rendering fallback when active eye input is unavailable.
- Frame-reuse, asynchronous timewarp, and DFR/FFR alignment fixes intended to reduce flicker and visible instability.

The main implementation areas are:

- `alvr/client_openxr/` — eye-tracking acquisition and client interaction.
- `alvr/packets/` — eye-tracking state and tracking data transport.
- `alvr/server_core/src/tracking/` — server-side eye-tracked foveated rendering.
- `alvr/server_openvr/cpp/platform/win32/FFR.cpp` — server-side foveated encoding parameters.
- `alvr/graphics/resources/stream.wgsl` — client-side inverse foveation and frame rendering.

The branch also contains exploratory telemetry and motion-prediction work from later experiments. Those changes are retained in the history, but they are not presented as production-ready or actively maintained features.

## Status and limitations

This is a reference implementation and an invitation for further development. The original developer currently has no VR headset available for continued hardware validation or maintenance. Porting the DFR changes to a current upstream ALVR revision, validating them on modern headsets, and preparing an upstream contribution will require community help.

Use this repository to inspect the implementation, reproduce the research direction, or extract individual ideas for upstream integration. Expect substantial work when adapting it to current ALVR APIs, protocols, graphics code, and headset runtimes.

## Building

The minimum Rust version is 1.82. Windows builds require the MSVC C++ toolchain and Windows SDK. Android builds additionally require Java 17, Android SDK platform 32, Android NDK r26, and the `aarch64-linux-android` Rust target.

The server core's BCMP predictor requires libtorch 2.4.0 when that feature is enabled. Set `LIBTORCH` to the libtorch installation before building.

From the repository root:

```powershell
cargo xtask prepare-deps --platform windows
cargo xtask build-streamer --release --gpl
cargo xtask build-launcher --release
```

To build the Android client:

```powershell
cargo xtask prepare-deps --platform android --ci
cargo xtask build-client --release
```

To compile all workspace test targets without running hardware-dependent tests:

```powershell
cargo test --workspace --no-run
```

Build outputs are written below `build/` and `target/`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
