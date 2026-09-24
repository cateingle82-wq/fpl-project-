/**
 * Persisted app settings: the FastAPI backend's base URL and your FPL
 * team ID. Both live in AsyncStorage (per-device, not synced anywhere)
 * so you only type them once.
 *
 * Default API_BASE_URL is "http://localhost:8000" — this ONLY resolves
 * correctly from the iOS Simulator (which shares the host Mac's network
 * namespace). It will NOT work from:
 *   - a physical phone: use the dev machine's LAN IP instead, e.g.
 *     "http://192.168.1.42:8000" (find it with `ipconfig getifaddr en0`
 *     on the Mac running `uvicorn api:app --host 0.0.0.0`)
 *   - the Android emulator: use "http://10.0.2.2:8000" (its special
 *     alias for the host machine's loopback)
 * Change it in the Settings screen — see src/app/settings.tsx.
 */
import AsyncStorage from "@react-native-async-storage/async-storage";

const KEYS = {
  apiBaseUrl: "fpl.apiBaseUrl",
  teamId: "fpl.teamId",
  horizon: "fpl.horizon",
} as const;

export const DEFAULT_API_BASE_URL = "http://localhost:8000";
// Matches fpl_stage0.HORIZON's own default — NOT the same as the "3" the
// very first version of this screen hardcoded in its fetch calls, which
// was an arbitrary placeholder, not a real backend default.
export const DEFAULT_HORIZON = 5;

export async function getApiBaseUrl(): Promise<string> {
  const stored = await AsyncStorage.getItem(KEYS.apiBaseUrl);
  return stored ?? DEFAULT_API_BASE_URL;
}

export async function setApiBaseUrl(url: string): Promise<void> {
  await AsyncStorage.setItem(KEYS.apiBaseUrl, url.trim());
}

export async function getTeamId(): Promise<string> {
  const stored = await AsyncStorage.getItem(KEYS.teamId);
  return stored ?? "";
}

export async function setTeamId(teamId: string): Promise<void> {
  await AsyncStorage.setItem(KEYS.teamId, teamId.trim());
}

export async function getHorizon(): Promise<number> {
  const stored = await AsyncStorage.getItem(KEYS.horizon);
  const n = stored ? Number(stored) : DEFAULT_HORIZON;
  return Number.isFinite(n) && n >= 1 && n <= 8 ? n : DEFAULT_HORIZON;
}

export async function setHorizon(horizon: number): Promise<void> {
  await AsyncStorage.setItem(KEYS.horizon, String(horizon));
}
