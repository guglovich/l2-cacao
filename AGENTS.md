# dxvk-cacao-build — AGENTS.md

## ТЕКУЩИЙ СТАТУС (07.11.2026): **AO РАБОТАЕТ В ИГРОВОМ МИРЕ — ВПЕРВЫЕ (подтверждено пользователем)**

Логин, лобби выбора персонажа и игровой мир дают привязанное к геометрии
затенение. В мире пока слабое — тюнинг под дистанции применён (radius 6.0,
strength 0.6), ожидает проверки.

**Ветка:** `cacao-3.1`. Стек фиксов (все в git):
`bf69b534` — D3D9-конвенция depth + неадаптивные варианты генерации;
`5f91ccf1` — композит по координатам источника;
`46d25aad` — триггер только по UI + SDK-фадаут до 1600 юнитов;
`c06f60d4` — mid-frame композит (AO в backbuffer до UI, UI не темнеет);
`e4184be8` — ре-арминг цепочки (L2 рендерит объекты ПОСЛЕ первого UI-орто);
`3666da30` — милстоун-документация.

**Карта стеков и обходных путей:** `docs/DX9_VULKAN_WINE_PATHS.md`.

**Ключевое знание о кадре L2 (важно для будущих правок):**
мир: мировая проекция → террейн → UI-орто(1) → объекты/персонажи со своей
мировой проекцией → UI-орто(2) → Present. Цепочка стреляет на UI-орто и
ре-армируется на поздних мировых проекциях (до 3 за Present). Вода
([6.31]/[4.16]) — off-aspect, триггером не является.

**Рабочий конфиг** (`/Games/ElmoreX3/system/dxvk.conf`):
```ini
d3d9.cacaoEnable=true, cacaoQuality=3, cacaoRadius=6.0, cacaoStrength=0.6,
cacaoShadowClamp=0.94, cacaoDarkMask=0.3, cacaoMidComposite=true,
cacaoHotkey=Insert, dxvk.enableDescriptorHeap=False
```

**Инструменты:** `tools/l2bg_run.sh` (headless-прогон, не трогает
пользовательскую сессию), `l2bg_world_watch.sh` (авто-анализ мировых дампов
→ `tmp/bg/world_watch_report.txt`), `l2bg_world_dump_analysis.py`,
`l2bg_sim_strength.py` (офлайн-симуляция strength), `l2bg_ab_analysis.py`,
`dxvk/tools/compile_cacao_mid.sh` (шейдеры mid-композита).

**Автоинструменты:** `tools/l2bg_run.sh` (headless-прогон; не трогает
пользовательскую сессию), `l2bg_world_watch.sh` (вотчер дампов),
`l2bg_world_dump_analysis.py` (анализ дампов), `l2bg_sim_strength.py`
(офлайн-симуляция strength), `l2bg_ab_analysis.py` (A/B по кадрам),
`l2bg_muter.sh` (мьют аудио).

**Подтверждение (headless-прогон 03.11, логин-сцена):**
- depth читается корректно: viewZ 207..1446 (структура сцены — стволы, лианы, папоротники);
- AO структурный и привязан к геометрии: mean 0.952, p5 0.855, p1 0.808,
  13.9% пикселей < 0.9 (контактные тени в зарослях, чистое небо/тропа);
- осцилляции НЕТ: дамп-слоты A и B идентичны;
- мерцания НЕТ: 30-кадровый burst, макс. межкадровая дельта 0.93;
- композит чистый (скриншоты без пятен/инверсий/чёрных кадров).

### ПЯТЬ корневых багов (полная история цикла «пятна ↔ ноль ↔ мерцание»)

1. **VK_EXT_descriptor_heap конфликт** (DXVK 3.0+ heap-модель по умолчанию).
   Классические descriptor sets CACAO рядом с heap-биндингом молча возвращают
   нули → AO=0 → эффект невидим. **Фикс:** `dxvk.enableDescriptorHeap = False`.

2. **Инвертированный композит** (исторический): CACAO пишет ВИДИМОСТЬ
   (1.0 = освещено), композит `color.rgb *= ao;`. Исправлено 06.09.

3. **Масштаб радиуса**: `worldRadius = cacaoRadius * near(≈10)`. Значения
   200–300 давали тысячи юнитов → полная окклюзия → чёрный экран.

4. **ОШИБОЧНЫЙ «РЕВЕРС DEPTH» (исправлено 03.11, коммит bf69b534).** Вечерняя
   итерация 06.09 «исправила» константы распаковки на реверс
   (`(-mul, 1-add)`) на основании наблюдения «пустой буфер даёт viewZ=близь» —
   наблюдение было с НЕИНИЦИАЛИЗИРОВАННОГО буфера. DXVK хранит D3D9 depth
   БЕЗ реверса (0=near, 1=far; Clear пишет 1.0=far; реверс-вьюпорта для D3D9
   в коде нет). Доказательство дампом 29.10/03.11: реальная сцена читается
   как raw 0.995 равномерно (= viewZ ~1900, задник логина); реверс-константы
   распаковывали её в viewZ ~10 (у near-плоскости) → мусорные нормали →
   SSAO ~0 → ЧЁРНЫЙ КОМПОЗИТ на каждом кадре с реальным depth. Это был
   источник «пятен» и «мерцания в мир». Правильные константы:
   `viewZ(d) = mul/(add - d)`, mul=-proj[14], add=proj[10] — как было до 06.09.

5. **ADAPTIVE Q3 TEMPORAL FEEDBACK (исправлено 03.11).** Порт всегда
   диспатчил q3-вариант generate — это ADAPTIVE шейдер: его ветка
   (qualityLevel==3) подмешивает «base pass» через LoadBasePassSSAOPass из
   pong-буфера, а базовый пасс в порте не запускается — в pong лежит
   промежуточный blur ПРОШЛОГО кадра → межкадровая обратная связь →
   период-2 осцилляция (0.95<->0.015, мерцание всего экрана). Фикс: добавлены
   пайплайны по качеству (q0/q1/q2 SPIR-V; интерфейс дескрипторов идентичен
   q3 — проверено spirv-dis); Medium и High используют неадаптивный q2.
   Полный адаптивный пайплайн (q3_base + importance-пассы) НЕ портирован —
   при портировке снять ограничение в Apply() (выбор generatePass).

### Рабочий dxvk.conf (тюнинг 03.11.2026, ночь — после свипа и симуляции)

```ini
d3d9.cacaoEnable = true
d3d9.cacaoQuality = 3         # исполнится как неадаптивный q2 (см. баг 5)
d3d9.cacaoRadius = 4.0        # x near(≈10) → 40 юнитов
d3d9.cacaoStrength = 0.5      # x2=1.0: фон ~-3% яркости, контакты до -28%
d3d9.cacaoShadowClamp = 0.94
d3d9.cacaoDarkMask = 0.3      # защита запечённых теней (0.6-0.8 гасит эффект)
d3d9.cacaoHotkey = Insert
dxvk.enableDescriptorHeap = False   # ОБЯЗАТЕЛЬНО
```

Шкала силы (офлайн-симуляция по дампу, mean/p5 видимости): 0.25→0.952/0.855
(на грани видимости — «изменений не вижу»), **0.5→0.905/0.715 (финальный)**,
0.75→0.860/0.586, 1.0→0.814/0.464 (глобальный смыв). Композит AO подтверждён
end-to-end: `cacaoDebugFill=true` темнит кадр (серия 120 кадров 71→57.6);
живой A/B с переключением Insert даёт ступень −2.1 единицы яркости ровно в
кадре переключения. ВАЖНО: при debugFill дамп-PGM нулевые (ветка debugFill
делает return до кода дампа) — это не баг композита.

### Фоновые прогоны без помех пользователю (новые инструменты, 03.11)

- `tools/l2bg_run.sh start|shot|stop <имя>` — игра в headless gamescope
  (`--backend headless`, рендер на реальной RADV, экран пользователя не
  затрагивается). Скриншоты — через pipewire-стрим gamescope
  (`gst-launch-1.0 pipewiresrc target-object=<node> ... pngenc`).
  Аудио глушится `tools/l2bg_muter.sh` (мьютит sink-input l2.exe).
- `l2bg_run.sh shot` кладёт PNG в `tmp/bg/` + печатает среднюю яркость.
- Горячая клавиша из скриптов: окно игры на внутреннем Xwayland gamescope
  (DISPLAY=:1 при одном gamescope), активировать окно и слать
  `xdotool keydown Insert; sleep 0.6; xdotool keyup Insert`
  (короткий XTEST-клик может теряться). `--window` (XSendEvent) wine игнорирует.
- Диагностика-правило: **cacaoDump=true — источник истины** (PGM-дампы
  ao/depth/ssao). `cacaoDebugView` в blitter сейчас врёт (равномерный серый
  ~0.65 вместо реального AO — семплирует не тот источник; чинить при случае).
  Readback-пробы стабильны, но подозреваются в устаревании (kFrameLag-слайс);
  при расхождении верить дампу.

### Деплой (системный wine-префикс НЕ трогать)

- DLL живёт внешней копией: `/Games/ElmoreX3/system/d3d9.dll` (скрипт
  перезаписывает при каждом запуске — переживает чистку папки).
- Обычный запуск (на экране пользователя): `~/Projects/dxvk-cacao-build/run_cacao31.sh`.
- Фоновый (без помех): `tools/l2bg_run.sh start <имя>`.
- Откат на внутреннюю DLL префикса: `WINEDLLOVERRIDES="d3d9=b"`.

### Специфика порта на 3.1 (для будущих ребейзов)

- `spillRenderPass(false)` → `endCurrentPass(false)`; `m_rtLayouts` удалён —
  `notifyExternalDepthLayout()` оставлен no-op (per-image tracking + unified layouts).
- `cmdBindDescriptorSets` → `VkBindDescriptorSetsInfo` + `stageFlags` (binds2).
- `dxvk_graphics.*` держать девственными апстримными.
- Сабмодули включают dxbc-spirv (новый компилятор шейдеров, свой кэш в AppData).
- meson: опция `enable_tests` удалена в 3.1.
- d3d11_cacao.* удалены из сборки (сломанный эксперимент, лежит в backup-ветке).

### Открытые вопросы

- **В мире не проверено** (нужен логин персонажа): ожидание — контактные тени
  как на логине; метрика качества — отсутствие глобального затемнения.
- Полный адаптивный Q3 (q3_base + importance-пассы) не портирован — сейчас
  High исполняет q2. Визуальная разница мала, перф выгоду даст позже.
- `cacaoDebugView` в blitter показывает равномерный серый вместо AO — починить
  привязку при случае (диагностика сейчас через cacaoDump).
- Хоткей Insert работает (маркеры [HOTKEY] в cacao.log).
- GPU-нагрузка: в headless-логине ~30 fps ([PACE] 60 presents in ~1900 ms);
  полная метрика — только в игровом мире.

---

## Историческая часть (период 2.7.1, апрель 2026)

### Описание проекта

Форк DXVK v2.7.1 (commit `c3dd74be`) с интегрированным FidelityFX CACAO (Combined Adaptive Compute Ambient Occlusion) для D3D9.

**Цель:** Реализовать SSAO в D3D9 играх через Wine/DXVK путём модификации DXVK на уровне перехвата depth buffer и запуска compute shader pipeline.

**Целевая игра для тестирования:** Lineage 2 (Unreal Engine 2 mod, D3D9).

**Путь к проекту:** `/home/jinseoyeon/Projects/dxvk-cacao-build/`

---

## Методология разработки

### Используемые навыки и расширения

| Навык | Назначение |
|-------|-----------|
| **superpowers (using-superpowers)** | Дисциплинированный workflow — проверка skills перед любым действием, brainstorming перед изменениями, systematic-debugging для багов |
| **context7-mcp** | Актуальная документация по Vulkan, DXVK, SPIR-V, glslang. Использовать для проверки API, флагов, VUID ошибок |
| **basic-memory** | База знаний проекта. Сохранять результаты диагностики, архитектурные решения, найденные баги. Персистентность между сессиями |
| **basic-memory-rag** | Семантический поиск по сохранённым знаниям о Vulkan/DXVK/CACAO |
| **save-to-memory** | Фиксация важных решений и находок в конце сессии |

### Правила работы

1. **Перед любым изменением кода** — invoke `brainstorming` skill
2. **При отладке** — invoke `systematic-debugging` skill (не предлагать фиксы без изоляции бага)
3. **Перед завершением** — invoke `verification-before-completion` (запустить компиляцию/тест)
4. **При работе с Vulkan API** — проверить context7 для актуальной документации
5. **После диагностики** — сохранить в basic-memory

### Субагенты

| Агент | Когда использовать |
|-------|-------------------|
| `general-purpose` | Анализ кода, поиск багов, изучение файлов, веб-поиск, декомпиляция SPIR-V |
| `code-reviewer` | Review изменений перед коммитом |
| `permission-checker` | Проверка разрешений |
| `rag-trigger` | Поиск по базе знаний basic-memory |

**Правило:** Использовать субагентов для независимых задач параллельно. Для исследования кода — `general-purpose` вместо прямого чтения файлов.

---

## Результаты тестирования ReShade AO шейдеров через vkBasalt

Тестирование проведено для понимания ограничений vkBasalt и работы depth buffer с Lineage 2.

### DisplayDepth (диагностический шейдер)
- **Статус:** ✅ Работает корректно
- **Результат:** Depth buffer передаётся через vkBasalt в L2
- **Файл:** `/opt/reshade/shaders/DisplayDepth.fx`
- **Скриншот:** `displaydepth.png` — depth map (лево) + normal map (право)
- **Вывод:** vkBasalt может читать depth buffer из DXVK D3D9 → подход DXVK-форка валиден

### MXAO 4.0.2 EX
- **Статус:** ⚠️ Частично работает, артефакты
- **Результат:** AO затемнения видны в углах и у объектов, но чёрные квадратные артефакты по экрану
- **Причина артефактов:** Internal render targets (`MXAO_CullingTex`, `MXAO_NormalTex`) не инициализируются в vkBasalt
- **Depth API:** `ReShade::GetLinearizedDepth` (работает)
- **Плюс:** Не требует внешних текстур, не крашится
- **Скриншот:** `mxao.png`

### SSAO.fx (Marty McFly)
- **Статус:** ❌ Краш
- **Ошибка:** `vkBasalt err: couldn't open texture: /opt/reshade/textures/mcnoise.png`
- **Причина:** `mcnoise.png` отсутствует — noise-текстура для ротации семплов
- **Depth API:** `ReShade::GetLinearizedDepth` (работает корректно)

### HBAO.fx (NVIDIA)
- **Статус:** ❌ Краш + ошибка Vulkan loader
- **Ошибка:** `01cc:err:msvcrt:_wassert (L"!status && \"vkGetSwapchainImagesKHR\"", L"../wine/dlls/winevulkan/loader_thunks.c", 6747)`
- **Причины:**
  - `RandomNoiseB.dds` отсутствует
  - Собственный `depthSampler` не привязан к реальному depth buffer
  - Текстуры `frameTex2D`, `prevPassTex2D` пустые
- **Depth API:** Собственная `LinearizeDepth()` (не работает через vkBasalt)

### Выводы из тестирования

| Шейдер | Depth | Текстуры | Результат |
|--------|-------|----------|-----------|
| DisplayDepth | ✅ | ❌ не нужны | ✅ Работает |
| MXAO | ✅ | ✅ не нужны | ⚠️ Работал 1 раз, потом не запускался |
| SSAO | ✅ | ❌ mcnoise.png нет | ❌ Краш |
| HBAO | ❌ | ❌ RandomNoiseB.dds нет | ❌ Краш |

**Уточнение про MXAO:** Работал единожды при первом запуске, потом перестал. Причина — **сетевая блокировка ReShade** в MMO. ReShade намеренно отключает доступ к depth buffer при обнаружении сетевого трафика (защита от читов). Lineage 2 — MMO с постоянным обменом пакетами, поэтому после первого кадра depth buffer заблокирован. Это НЕ проблема шейдера — это античит-механизм ReShade.

**Ключевой вывод:** vkBasalt НЕ подходит для CACAO. ReShade FX шейдеры — fragment shaders с ограниченным multi-pass. CACAO — compute shader pipeline с barrier-ами, descriptor sets, и прямым Vulkan API. **Единственный путь для CACAO — DXVK-форк.**

---

## Анализ Marty's Mods Depth Buffer Guide

Источник: https://guides.martysmods.com/reshade/depth/

### Generic Depth Addon
ReShade автоматически выбирает depth buffer по эвристикам: draw calls, aspect ratio, разрешение, тип буфера. В D3D9 только одна поверхность depth-stencil.

### Copy depth buffer before clear operations
Движки очищают depth buffer через `ClearDepthStencilView` между этапами. ReShade перехватывает на Pre-Clear и копирует содержимое. **Критически важно для D3D9.**

### D3D9 специфика
| Особенность | Значение |
|-------------|----------|
| Depth buffer перевёрнут | `IS_UPSIDE_DOWN = 1` почти всегда |
| Одна поверхность depth-stencil | Ручной выбор только в режиме копирования |
| DXVK мерцание depth buffer | Известный баг DXVK 2.3.1+ (issue #3918, #4567) — закрыт как `not planned` |

### MXAO мерцание — корневая причина
**Сетевая блокировка ReShade** — главная причина. ReShade блокирует depth buffer при сетевой активности (античит). В MMO (Lineage 2) это происходит постоянно. Другие причины: фоновый софт (Logitech G Hub), несовместимость версий шейдера, dynamic resolution.

### Влияние на DXVK-CACAO
- DXVK мерцание depth buffer — известный баг, может влиять на стабильность CACAO pipeline
- `m_cacaoLastDepthImg` в DXVK-CACAO — правильный аналог `Copy depth before clear`
- Сетевой трафик L2 НЕ влияет на DXVK-CACAO (уровень DXVK, не ReShade)
- Альтернатива: ENB Series для D3D9 (встроенный AO, стабильнее в MMO)

---

## Архитектура проекта

### Структура файлов

```
dxvk-cacao-build/
├── dxvk/                    — подмодуль: оригинальный DXVK 2.7.1
├── dxvk-cacao.patch         — патч на ~522 строки (24 файла изменено)
├── src/
│   ├── d3d9/
│   │   ├── d3d9_cacao.cpp       — 1163 строки, основная реализация CACAO
│   │   ├── d3d9_cacao.h         — класс D3D9CacaoRenderer
│   │   ├── d3d9_cacao_log.h     — логгер в cacao.log
│   │   └── cacao_spirv/         — 26 .h файлов со SPIR-V бинарями
│   └── dxvk/hud/
│       └── dxvk_hud_cacao.h     — HUD-элемент статуса CACAO
├── build.sh                 — скрипт сборки (Arch Linux, 32-bit)
├── d3d9.dll                 — собранная DLL
├── validation.log           — лог валидации через Wine
└── run_validation.sh        — скрипт запуска L2.exe с VK_LAYER
```

### Как работает доступ к depth buffer

1. **Перехват в SetDepthStencilSurface** (`d3d9_device.cpp`):
   ```cpp
   if (ds != nullptr)
     m_cacaoLastDepthImg = ds->GetCommonTexture()->GetImage();
   ```
   Игра часто вызывает `SetDepthStencilSurface(nullptr)` перед Present (UI-проходы), поэтому `m_state.depthStencil` может быть null. Сохраняется последняя ненулевая глубина.

2. **Кэширование матрицы проекции** (`d3d9_device.cpp` строки ~7657-7664):
   ```cpp
   const Matrix4& proj = m_state.transforms[GetTransformIndex(D3DTS_PROJECTION)];
   // Reject identity matrix
   if (proj[0][0] != 1.0f || proj[1][1] != 1.0f || proj[3][2] != 0.0f) {
     m_cacaoLastProj    = proj;
     m_cacaoHasValidProj = true;
   }
   ```

3. **UBO константы** (`d3d9_cacao.cpp` строки ~560-590):
   ```cpp
   depthLinearizeMul = -proj[14];      // = (far*near)/(far-near)
   depthLinearizeAdd = proj[10];       // = far/(far-near)
   tanHalfFOVX = 1.0f / proj[0];
   tanHalfFOVY = 1.0f / proj[5];
   NDCToViewMul[0] = tanHalfFOVX * 2.0f;
   NDCToViewMul[1] = tanHalfFOVY * -2.0f;  // Y-flip для D3D9
   NDCToViewAdd[0] = -tanHalfFOVX;
   NDCToViewAdd[1] = +tanHalfFOVY;
   ```

4. **Compute pipeline** — 8 Vulkan compute passes:
   | # | Pass | Описание |
   |---|------|----------|
   | 1 | PREPARE_DEPTHS | Деинтерливинг depth + мипы (4 слоя, half-res) |
   | 2 | PREPARE_NORMALS | Нормали из depth (half-res) |
   | 3 | GENERATE_Q3 | Основной SSAO (Q3 non-adaptive) |
   | 4-7 | EDGE_BLUR x4 | 4 прохода двустороннего blur (ping-pong) |
   | 5 | APPLY | Апсемпл до full-res, запись AO в m_aoFinal (R16G16B16A16, R канал) |

5. **Композитинг** (`dxvk_present_common.glsl`):
   ```glsl
   if (c_composite_ao) {
     float ao = texelFetch(s_ao, coord, 0).r;
     color.rgb *= ao;  // <-- ПОТЕНЦИАЛЬНАЯ ПРОБЛЕМА: нет преобразования
   }
   ```

### Конфигурация (dxvk.conf)

```
d3d9.cacaoEnable = true
d3d9.cacaoQuality = 2        # 0=Lowest, 1=Low, 2=Medium, 3=High
d3d9.cacaoRadius = 1.2       # радиус в мировых единицах
d3d9.cacaoStrength = 1.0     # сила теней, 0-2
d3d9.cacaoShadowClamp = 0.98
d3d9.cacaoHotkey = Insert    # toggle on/off
d3d9.cacaoDebugFill = false  # заполнить AO серым 0.5 для тестирования композитинга
d3d9.cacaoDebugView = false  # показать raw AO как grayscale
```

---

## Критические баги (из validation.log)

### Баг #1: Binding 7 не объявлен в layout (КРИТИЧЕСКИЙ)

Шейдер `ImportanceMapA` требует binding 7 (`g_ImportanceMapPong` — sampled image), но layout `PASS_IMPORTANCE_MAP_A` в `d3d9_cacao.cpp` создаётся без binding 7.

```
VUID-VkComputePipelineCreateInfo-layout-07988:
pCreateInfos[0].stage SPIR-V uses descriptor [Set 0, Binding 7]
but the binding was not declared in VkPipelineLayoutCreateInfo::pSetLayouts[0]
```

**Результат:** шейдер читает из необъявленной привязки → undefined behavior → pipeline может не работать.

**Файл для исправления:** `src/d3d9/d3d9_cacao.cpp` ~строка 330, layout для `PASS_IMPORTANCE_MAP_A`.

**Решение (из context7 Vulkan docs):** Добавить `VkDescriptorSetLayoutBinding` для binding 7:
```cpp
VkDescriptorSetLayoutBinding binding7 = {};
binding7.binding = 7;
binding7.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
binding7.descriptorCount = 1;
binding7.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
// Добавить в массив bindings[] для PASS_IMPORTANCE_MAP_A
```

### Баг #2: vulkanMemoryModelDeviceScope не включена (КРИТИЧЕСКИЙ)

SPIR-V использует `OpAtomicIAdd` с `Device` memory scope, но фича не включена при создании Vulkan device.

```
VUID-RuntimeSpirv-vulkanMemoryModel-06265:
SPIR-V uses Device memory scope, but vulkanMemoryModelDeviceScope feature was not enabled.
%110 = OpAtomicIAdd
```

**Результат:** атомарные операции работают некорректно → счётчики ломаются.

**Файл для исправления:** `dxvk/src/dxvk/dxvk_adapter.cpp` — включить `vulkanMemoryModelDeviceScope` в `VkPhysicalDeviceVulkan12Features`.

**Решение (из context7 Vulkan docs):** Добавить в pNext chain VkDeviceCreateInfo:
```cpp
VkPhysicalDeviceVulkan12Features features12 = {};
features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
features12.vulkanMemoryModel = VK_TRUE;
features12.vulkanMemoryModelDeviceScope = VK_TRUE;
// Включить в pNext chain VkDeviceCreateInfo
```

**Примечание:** `vulkanMemoryModelDeviceScope` требует `vulkanMemoryModel = VK_TRUE` (оба флага нужны). RADV (AMD) поддерживает оба флага — проверено через `vkGetPhysicalDeviceFeatures2`.

### Баг #3: AO композитинг без преобразования (ВЫСОКИЙ)

`dxvk_present_common.glsl` строка ~60:
```glsl
color.rgb *= ao;
```

CACAO записывает raw occlusion в R-канал. Если AO value > 1 — сцена **осветлится** вместо затемнения. В оригинальном FFX SDK: `color *= max(0.0, 1.0 - ao * strength)`.

**Файл для исправления:** `dxvk/src/dxvk/shaders/dxvk_present_common.glsl`.

### Баг #4: CACAO Apply не запускается (СРЕДНИЙ)

В validation.log нет ни одного сообщения `[APPLY]` или `[FRAME] TryApply: RUNNING`. Причина: скорее всего `m_cacaoHasValidProj` никогда не становится `true` (identity projection check слишком строгий) или `m_cacaoLastDepthImg` всегда `nullptr`.

**Файл для отладки:** `src/d3d9/d3d9_device.cpp` строки ~7659-7664.

### Дополнительные ошибки в validation.log

- `VUID-VkRenderingAttachmentInfo-imageView-10780` — feedback loop image создан без `VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT` (ошибка DXVK, не CACAO)
- Crash в конце лога: `vkResetCommandPool` assertion failed (следствие незавершённых compute-операций)

---

## Состояние индустрии (на апрель 2026)

### vkBasalt — depth buffer

| Параметр | Статус |
|----------|--------|
| `depthCapture = on` существует | Да |
| Работает стабильно | Нет |
| Работает на NVIDIA | Частично, с багами |
| Работает на AMD/Intel | Практически нет |
| Правильно выбирает depth buffer | Нет, не всегда |
| RTGI / SSAO / AO работают | Нет |
| Проект активно развивается | Нет, заброшен ~2 года |

**Вывод:** vkBasalt НЕ решил проблему доступа к depth buffer. Параметр экспериментальный и сломан на большинстве GPU.

### DXVK — depth buffer

- **DXVK НЕ предоставляет прямой доступ к depth buffer** — архитектурная позиция doitsujin
- Issue #2708 и #3816 закрыты как "out of scope"
- Depth buffer — internal `VkImage` с `VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT` без `VK_IMAGE_USAGE_SAMPLED_BIT`
- Для чтения из шейдеров нужен `VK_IMAGE_USAGE_SAMPLED_BIT` и создание SRV

### Proton / vkd3d-proton

- Те же ограничения что и DXVK
- Никаких изменений по экспорту depth buffer в Proton Experimental
- Valve фокусируется на FSR и Frame Generation, не на depth buffer

### ReShade + DXVK

- ReShade Vulkan layer + DXVK: работает, но с overhead ~5-10% из-за copy
- ReShade D3D11 injector + DXVK: работает частично, depth capture ненадёжен
- Нативные Vulkan игры: полная поддержка depth buffer

### Вывод

**Единственный рабочий подход:** модификация DXVK для получения depth buffer на уровне `SetDepthStencilSurface` (ваш подход). vkBasalt, ReShade, Proton — все имеют те же или худшие проблемы.

---

## FidelityFX CACAO SDK

### Источники шейдеров

- **Standalone:** https://github.com/GPUOpen-Effects/FidelityFX-CACAO (v1.2, апрель 2021)
- **В составе SDK:** https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK (версия 1.1.4, последняя с CACAO)
- **Документация:** https://gpuopen.com/manuals/fidelityfx_sdk/techniques/combined-adaptive-compute-ambient-occlusion/

**Важно:** FidelityFX SDK 2.0 (август 2025) **больше не включает CACAO** — перешёл на ML-технологии. Для сверки шейдеров брать SDK 1.1.4 или standalone репозиторий.

### HLSL entry points

| Проход | Entry Point |
|--------|-------------|
| Prepare глубины | `FFX_CACAO_PrepareNativeDepthsAndMips` |
| Prepare нормалей | `FFX_CACAO_PrepareNativeNormalsFromInputNormals` |
| Генерация SSAO | `FFX_CACAO_GenerateSSAO` |
| Adaptive SSAO | `FFX_CACAO_AdaptiveSSAO` + `ImportanceMap` |
| Blur | `FFX_CACAO_Blur` |
| Application | `FFX_CACAO_Apply` |
| Upsample | `FFX_CACAO_BilateralUpsample` |

### Статус исходников в проекте

**В проекте остались только .h файлы со SPIR-V бинарями** (`src/d3d9/cacao_spirv/`). Исходные GLSL/HLSL утеряны на этапе ранней сборки. Шейдеры скомпилированы из оригинального FFX CACAO SDK и математически корректны.

**Проблема не в шейдерах** — проблема в Vulkan setup коде (`d3d9_cacao.cpp`).

---

## Доступные инструменты

### Установленные на системе

| Инструмент | Путь | Версия |
|------------|------|--------|
| clangd | /usr/bin/clangd | 20.1.8 |
| clang++ | /usr/bin/clang++ | 20.1.8 (libc++) |
| glslangValidator | /usr/bin/glslangValidator | — |
| spirv-cross | /usr/bin/spirv-cross | — |
| spirv-dis | /usr/bin/spirv-dis | — |
| cmake | /usr/bin/cmake | — |
| Vulkan SDK | ~/.local/share/VulkanSDK/1.4.341.0/ | 1.4.341.0 |

### GPU и драйвер

- **GPU:** AMD Radeon RX 9060 XT (RADV, NAVI24)
- **Драйвер:** mesa-vulkan-radeon 25.3.7
- **Vulkan:** 1.4.344

### LSP статус

**Настроено:**
- `.clangd` создан в корне проекта
- `compile_commands.json` → symlink на `dxvk/build/compile_commands.json`
- clangd 20.1.8 установлен

**Для активации в редакторе:**
- Vim + coc.nvim: `CocInstall coc-clangd`
- Neovim + nvim-lspconfig: `lspconfig.clangd.setup{}`
- VSCode: установить clangd extension

---

## Рекомендуемые субагенты для работы

| Агент | Когда использовать |
|-------|-------------------|
| `general-purpose` | Анализ кода, поиск багов, изучение файлов, веб-поиск |
| `code-reviewer` | Review изменений перед коммитом |
| `permission-checker` | Проверка разрешений на запись в /Media |
| `rag-trigger` | Поиск по базе знаний если есть заметки по Vulkan/DXVK |

### Skills

| Skill | Когда использовать |
|-------|-------------------|
| `brainstorming` | Перед началом нового этапа разработки |
| `systematic-debugging` | При отладке compute pipeline, SPIR-V, Vulkan ошибок |
| `verification-before-completion` | Перед утверждением что баг исправлен |
| `test-driven-development` | При написании новых compute shaders |

---

## Команды сборки

```bash
# Сборка через build.sh (Arch Linux, 32-bit)
cd /home/jinseoyeon/dxvk-cacao-build/
./build.sh

# Ручная сборка
cd dxvk
meson --cross-file build-win32.txt --buildtype release -Denable_tests=false build
ninja -C build
```

---

## Рекомендации по приоритету исправлений

1. **Баг #1** — добавить binding 7 в layout `PASS_IMPORTANCE_MAP_A` → это разблокирует запуск compute passes
2. **Баг #2** — включить `vulkanMemoryModelDeviceScope` → атомарные операции заработают
3. **Баг #3** — исправить композитинг: `color.rgb *= max(0.0, 1.0 - ao * strength)` → AO станет видимым
4. **Баг #4** — проверить почему `m_cacaoHasValidProj` не становится true → добавить логирование
5. **Тест** — включить `cacaoDebugFill = true` для проверки композитинга
