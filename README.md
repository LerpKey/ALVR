# PICO 4 Enterprise Eye-Tracked Dynamic Foveated Rendering for ALVR

An experimental [ALVR](https://github.com/alvr-org/ALVR) fork for **gaze-contingent dynamic foveated rendering (DFR)** on the **PICO 4 Enterprise (PICO 4E) eye-tracking headset**. The implementation moves the high-quality foveal region with the user's gaze instead of keeping it fixed at the image center.

This project targets the eye-tracking PICO 4E configuration specifically. Standard PICO 4 models and other headsets are not validated for the eye-tracked DFR path. When valid eye input is unavailable, the implementation can fall back to fixed foveated rendering.

It is not a current upstream ALVR release and should not be expected to be compatible with the latest upstream code or headset runtimes.

## What this project provides

- PICO 4E eye-gaze acquisition through the Android OpenXR client.
- Per-eye gaze data transport from the headset to the ALVR streamer.
- Gaze-contingent server-side foveated encoding with a moving foveal region.
- Client-side inverse foveation to reconstruct the streamed image.
- Explicit `Unsupported`, `Standby`, and `Active` eye-tracking states.
- Fixed foveation fallback when active eye input is unavailable.
- Optional telemetry for eye tracking, pose, encoding, and network measurements.

This is a reference implementation for people looking for an ALVR-compatible PICO 4E DFR starting point. It is not a maintained production release and has not been validated against current upstream ALVR versions.

## Fork provenance

The code lineage starts from the official ALVR `master` branch at commit [`1f0ba243`](https://github.com/alvr-org/ALVR/commit/1f0ba243c0fd46cb0b54e2e54d46719a45fc1664), dated 2025-05-09:

> Remove PrefersNonDefaultGPU from desktop files (#2812)

The first local development snapshot was created directly on top of that upstream commit. This repository therefore represents an older ALVR baseline, not a fork of the current upstream tip.

The public history intentionally contains only two source-focused commits: the DFR implementation and the later telemetry/data-collection additions. Intermediate experiment commits and private materials are not included.

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

The branch also contains telemetry and motion-prediction additions. Those changes are not presented as production-ready or actively maintained features.

## Hardware and experiment profile

The eye-tracking development and the available logs are centered on **PICO 4 Enterprise (PICO 4E)**. The target desktop path is Windows with SteamVR, the Windows OpenVR driver, NVIDIA hardware encoding through NVENC, and the Direct3D 11 server renderer. The exact PC CPU, GPU model, memory, router model, PICO firmware version, and OpenXR runtime version were not recorded in the development logs; this repository does not claim those details.

The recommended experiment network uses a 5 GHz wireless connection for the headset and Ethernet for the streamer PC on the same local network. The exact network hardware and measured bandwidth are not part of the recorded baseline.

## Status and limitations

This is a reference implementation and an invitation for further development. Historical validation was centered on PICO 4E, but this repository does not claim ongoing hardware validation or broad headset compatibility. Porting the DFR changes to a current upstream ALVR revision, validating them on modern headsets, and preparing an upstream contribution will require community help.

Use this repository to inspect the implementation, reproduce the development direction, or extract individual ideas for upstream integration. Expect substantial work when adapting it to current ALVR APIs, protocols, graphics code, and headset runtimes.

## Building

The minimum Rust version is 1.82. Windows builds require the MSVC C++ toolchain and Windows SDK. Android builds additionally require Java 17, Android SDK platform 32, Android NDK r26, and the `aarch64-linux-android` Rust target.

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
