#!/bin/bash
# Компиляция CACAO GLSL (родные исходники FFX + DXVK-конвенции биндингов)
# -> cacao_spirv/dxvk_cacao2_*.h.
#
# Схема биндингов (set 1 — ресурсы, set 0 — sampler heap, см. cacao_glsl/):
#   prepare:    0 depthIn, 1 normalIn, 2..5 depthMips[4], 6 deintD, 7 deintN, 8 CB
#   generate:   0 loadCnt, 1..4 SRV, 5 UAV ping, 6 CB
#   blur/apply: 0 SRV, 1 UAV, 2 CB
#   importance: 0..2 SRV, 3..5 UAV, 6 CB
#   clear:      0 UAV, 1 CB
#
# Использование: bash dxvk/tools/compile_cacao2.sh  (из корня проекта)
set -e
PROJ="$(dirname "$(dirname "$0")")"   # dxvk/
SDKGPU="$PROJ/../FidelityFX-SDK-v1.1.4/sdk/include/FidelityFX/gpu"
SDKGPUCACAO="$SDKGPU/cacao"
GLSL="$PROJ/src/d3d9/cacao_glsl"
OUT="$PROJ/src/d3d9/cacao_spirv"

PASSES="cacao_prepare_depths cacao_prepare_normals cacao_generate_q0 \
cacao_generate_q1 cacao_generate_q2 cacao_blur8 cacao_apply \
cacao_importance_a cacao_importance_b cacao_importance cacao_clear_counter"

mkdir -p "$OUT"
for BASE in $PASSES; do
  glslangValidator --quiet --target-env vulkan1.3 \
    -DFFX_GPU -DFFX_GLSL \
    -I"$GLSL" -I"$SDKGPU" -I"$SDKGPU/cacao" \
    "$GLSL/${BASE}.comp" -o "/tmp/${BASE}.spv"
  spirv-val "/tmp/${BASE}.spv"
  echo "$BASE: $(stat -c %s /tmp/${BASE}.spv) bytes ok"
done

python3 - <<'PYEOF'
import struct, glob, os
for f in sorted(glob.glob('/tmp/cacao_*.spv')):
    name = os.path.basename(f)[:-4]
    blob = open(f,'rb').read()
    assert len(blob) % 4 == 0
    words = struct.unpack(f'<{len(blob)//4}I', blob)
    lines = []
    for i in range(0, len(words), 6):
        lines.append('    ' + ', '.join(f'0x{w:08x}' for w in words[i:i+6]) + ',')
    hdr = (f'// {name}: native FFX GLSL + DXVK binding conventions\n'
           f'const uint32_t {name.upper()}_SPV[] = {{\n' + '\n'.join(lines) + '\n};\n')
    out = f'/home/jinseoyeon/Projects/dxvk-cacao-build/dxvk/src/d3d9/cacao_spirv/{name}_pass.h'
    open(out,'w').write(hdr)
    print(name, len(words), 'words ->', os.path.basename(out))
PYEOF
