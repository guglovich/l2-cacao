# l2-cacao

**DXVK 3.1 fork with AMD FidelityFX CACAO (screen-space ambient occlusion) for
D3D9 games under Wine — built for and verified on Lineage 2 (UE2 engine).**

> Русская версия ниже.

## What is this

A fork of [DXVK](https://github.com/doitsujin/dxvk) (base v3.1) that injects
the full FidelityFX CACAO compute pipeline into the D3D9 device: it captures
the game's depth buffer, runs the SSAO chain (prepare → generate → edge-aware
blur → bilateral apply) and bakes the result into the back buffer before the
game's UI draws. Verified working on Lineage 2 (login, character select, and
the game world) under Wine on AMD RADV; should work on any Vulkan 1.3 GPU.

## Why a fork

A D3D9 game cannot reach its own depth buffer through the API, and DXVK does
not expose it either. The depth is captured at the DXVK driver level, which
makes CACAO's multi-pass compute chain possible — ReShade/vkBasalt-style
injections cannot do this reliably for D3D9 (known depth access/flicker
limitations).

## L2-specific problems solved (see git log for details)

| Problem | Root cause | Fix |
|---|---|---|
| Black spots / all-black frames | wrong depth unpack convention (D3D9: 0=near, 1=far) | constants restored, verified by full-buffer dumps |
| Fast flicker (period-2) | adaptive Q3 shader folds in the previous frame's buffer | non-adaptive q0/q1/q2 variants per quality |
| No AO in the world | L2 renders terrain, a UI fragment, then pawns — the chain fired on a depth holding only the sky dome | chain triggers at UI transforms, re-arms on later world projections (max 3 per present) |
| Weak AO in the world | fadeout killed everything past 800 units | SDK-style fadeout, zero at radius×40 |
| UI darkening | present-time composite covers UI | mid-frame composite bakes AO into the back buffer before the UI draws |
| In-game slowdowns | 275 MB log spam from alternating projections + legacy binding model | logger rate limiting + size cap; descriptor bindings migrated to the DXVK heap model (native FFX GLSL recompiled with DXVK conventions) |

## Build (32-bit D3D9)

```bash
meson setup --cross-file build-win32.txt --buildtype release build32
meson compile -C build32
# -> build32/src/d3d9/d3d9.dll
```

Requires meson, ninja, mingw-w64 (posix threads), glslangValidator.

## Configuration

`dxvk.conf` next to the game executable:

```ini
d3d9.cacaoEnable      = true
d3d9.cacaoQuality     = 3        # Medium/High run the non-adaptive q2 variant
d3d9.cacaoRadius      = 6.0      # world units ×near → contact shadow size
d3d9.cacaoStrength    = 0.6      # ×2 = internal shadow strength
d3d9.cacaoShadowClamp = 0.94
d3d9.cacaoDarkMask    = 0.3      # suppress AO on already-dark (baked) pixels
d3d9.cacaoMidComposite = true    # bake AO before UI draws (recommended)
d3d9.cacaoHotkey      = Insert   # toggle for A/B comparison
d3d9.cacaoDump        = false    # full-buffer PGM dumps for diagnostics
```

Deploy `d3d9.dll` next to the game exe and run with
`WINEDLLOVERRIDES="d3d9=n,b"`. Do not replace wine's own `dxgi.dll`
(L2 server integrity checks).

## Repository layout

- `src/d3d9/d3d9_cacao.{cpp,h}` — the CACAO renderer: Vulkan compute chain,
  mid-frame composite, telemetry
- `src/d3d9/cacao_glsl/` — the FFX GLSL passes with DXVK binding conventions
  (resources in set 1, samplers via the sampler heap at set 0)
- `src/d3d9/cacao_spirv/` — compiled SPIR-V headers
  (`tools/compile_cacao2.sh` reproduces them from the GLSL)
- `docs/DX9_VULKAN_WINE_PATHS.md` — the D3D9→Vulkan→Wine path map:
  limiters and workarounds
- `AGENTS.md`, `DEVLOG.md` — engineering notes and the full debug history

## Status

Working: login / character select / game world, stable (no flicker,
no artifacts), heap binding model (dxvk defaults), telemetry with
self-throttling logger.

Known limitations: adaptive Q3 quality is not ported (Medium/High run the
non-adaptive q2 variant); UE2 baked lightmaps are compensated by darkMask,
not re-lit.

---

# l2-cacao (рус.)

Форк DXVK 3.1 с внедрённым AMD FidelityFX CACAO (SSAO) для D3D9-игр под
Wine, собран и проверен на Lineage 2 (UE2). Глубина буфера захватывается на
уровне драйвера DXVK, CACAO считается в Vulkan compute-пайплайне и
запекается в backbuffer до отрисовки UI.

Сборка и конфигурация — выше. Подробная история отладки (пять корневых
багов: конвенция depth, адаптивный фидбек, триггер кадра, фадаут, лог-спам)
— в `DEVLOG.md`, карта путей DX9→Vulkan→Wine — в `docs/`.
