#!/bin/bash
# Компиляция шейдеров mid-frame композита -> cacao_spirv заголовки.
# Запуск из корня dxvk/: bash tools/compile_cacao_mid.sh
set -e
cd "$(dirname "$0")/.."
for SH in dxvk_cacao_mid_vert.vert dxvk_cacao_mid_frag.frag; do
  BASE=${SH%.*}
  glslangValidator --quiet --target-env vulkan1.3 -Isrc/dxvk/shaders \
    src/dxvk/shaders/$SH -o /tmp/${BASE}.spv
  spirv-val /tmp/${BASE}.spv
done
python3 - <<'PYEOF'
import struct
for name in ('dxvk_cacao_mid_vert', 'dxvk_cacao_mid_frag'):
    blob = open(f'/tmp/{name}.spv','rb').read()
    assert len(blob) % 4 == 0, (name, len(blob))
    words = struct.unpack(f'<{len(blob)//4}I', blob)
    lines = []
    for i in range(0, len(words), 6):
        chunk = ', '.join(f'0x{w:08x}' for w in words[i:i+6])
        lines.append('    ' + chunk + ',')
    ext = 'vert' if 'vert' in name else 'frag'
    body = '\n'.join(lines)
    hdr = f'// {name}: compiled from dxvk_cacao_mid_{ext} (glslangValidator, vulkan1.3)\nconst uint32_t {name.upper()}_SPV[] = {{\n{body}\n}};\n'
    open(f'src/d3d9/cacao_spirv/{name}_pass.h','w').write(hdr)
    print(name, len(words), 'words -> src/d3d9/cacao_spirv/')
PYEOF
