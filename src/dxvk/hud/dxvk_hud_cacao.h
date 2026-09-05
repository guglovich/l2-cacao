#pragma once

#include <atomic>

namespace dxvk::hud {

  /// Set to true once the CACAO renderer is fully initialized.
  extern std::atomic<bool> g_cacaoActive;

  /// Current enable/disable state (toggled by hotkey).
  extern std::atomic<bool> g_cacaoEnabled;

  /// Quality level 0-3 (Lowest/Low/Medium/High).
  extern std::atomic<int>  g_cacaoQuality;

}
