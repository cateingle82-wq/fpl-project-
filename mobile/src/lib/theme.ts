/**
 * Shared design tokens — one place for color/spacing/radius/type so every
 * screen looks like part of the same app instead of independently
 * styled. Colors take inspiration from FPL's own purple/green/pink
 * palette without reproducing exact brand assets.
 */

export const colors = {
  primary: "#37003c",       // deep purple — headers, primary buttons
  primaryLight: "#5c1a63",
  accent: "#00ff85",        // bright green — positive numbers, highlights
  danger: "#e90052",        // pink/red — errors, hits, warnings
  background: "#f5f5f8",
  surface: "#ffffff",
  border: "#e5e5ea",
  textPrimary: "#1c1c1e",
  textSecondary: "#6e6e73",
  textOnPrimary: "#ffffff",
} as const;

// One colour per FPL position — a common fantasy-football convention,
// used as small badges so a player's position reads at a glance instead
// of as another line of text.
export const positionColors: Record<string, string> = {
  GKP: "#f5a623",
  DEF: "#4a90d9",
  MID: "#00c37a",
  FWD: "#e94b4b",
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
} as const;

export const radius = {
  sm: 8,
  md: 12,
  lg: 16,
  pill: 999,
} as const;

export const type = {
  title: { fontSize: 22, fontWeight: "700" as const, color: colors.textPrimary },
  subtitle: { fontSize: 15, fontWeight: "700" as const, color: colors.textPrimary },
  body: { fontSize: 14, color: colors.textPrimary },
  caption: { fontSize: 12, color: colors.textSecondary },
  metricValue: { fontSize: 22, fontWeight: "800" as const, color: colors.primary },
  metricLabel: { fontSize: 11, color: colors.textSecondary, textTransform: "uppercase" as const, letterSpacing: 0.5 },
};

export const shadow = {
  shadowColor: "#000",
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.06,
  shadowRadius: 6,
  elevation: 2,
};
