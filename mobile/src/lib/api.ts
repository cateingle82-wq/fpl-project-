/**
 * Typed client for the FastAPI backend (../../api.py at the repo root).
 * Every shape here mirrors that file's actual response bodies exactly —
 * if you change a field there, change it here too; there's no shared
 * schema between the two languages to keep them honest automatically.
 */

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(baseUrl: string, path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${baseUrl}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    // A network-level failure (host unreachable, wrong URL, server not
    // running) throws a generic TypeError from fetch() with no useful
    // message on React Native — this is deliberately more actionable,
    // since "wrong base URL" is the single most likely first-run mistake.
    throw new ApiError(0, `Could not reach ${baseUrl} — check the API base URL in Settings ` +
      "and make sure the backend is running (uvicorn api:app --host 0.0.0.0).");
  }

  if (!res.ok) {
    // FastAPI's HTTPException bodies are always {"detail": "..."} —
    // matches api.py's own error responses exactly (see its
    // _fetch_or_400 helper and validation errors).
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // body wasn't JSON — fall back to statusText, already set above.
    }
    throw new ApiError(res.status, detail);
  }

  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Shapes — mirrors api.py's actual return dicts.
// ---------------------------------------------------------------------------

export type RiskLevel = "out" | "doubtful" | "impact_sub" | "fringe" | "ok";

export interface PlayerRisk {
  level: RiskLevel;
  detail: string | null;
}

export interface PlayerSummary {
  id: number;
  name: string;
  pos: string;
  team: string;
  price: number;
  xpts: number;
  risk: PlayerRisk;
  fixture: string;
  captain?: boolean;
}

export interface PlanWeek {
  week_offset: number;
  gw: number;
  captain: PlayerSummary;
  xi: PlayerSummary[];
  bench: PlayerSummary[];
  // Absent on week 0 (that's the top-level `transfers` block instead) —
  // present from week 1 on, describing the change from the PREVIOUS
  // week in the plan.
  transferred_in?: PlayerSummary[];
  transferred_out?: PlayerSummary[];
  hits?: number;
  free_transfers_available?: number;
}

export interface RecommendResponse {
  gw: number;
  horizon: number;
  objective: number;
  average_per_gw: number;
  transfers: {
    out: PlayerSummary[];
    in: PlayerSummary[];
    hits: number;
    hit_cost_paid: number;
  };
  captain: PlayerSummary;
  xi: PlayerSummary[];
  bench: PlayerSummary[];
  squad_cost: number;
  budget: number;
  plan: PlanWeek[];
}

export interface ChipScore {
  score: number | null;
  verdict: string;
}

export interface ChipsResponse {
  gw: number;
  horizon: number;
  bench_boost: { by_week: Record<string, number>; score: ChipScore };
  triple_captain: { by_week: Record<string, number>; score: ChipScore };
  wildcard: { gain: number; score: ChipScore };
  free_hit: { gain: number; score: ChipScore };
}

export interface SquadResponse {
  gw: number;
  player_ids: number[];
  bank: number;
  free_transfers: number;
}

export interface DeadlineResponse {
  gw: number;
  deadline_time: string | null;
}

// ---------------------------------------------------------------------------
// Requests — one function per api.py endpoint.
// ---------------------------------------------------------------------------

export interface SquadRequestBody {
  team_id?: number;
  manual_squad?: number[];
  horizon?: number;
  bank?: number;
  free_transfers?: number;
}

export function getHealth(baseUrl: string) {
  return request<{ status: string }>(baseUrl, "/health");
}

export function getDeadline(baseUrl: string) {
  return request<DeadlineResponse>(baseUrl, "/deadline");
}

export function getSquad(baseUrl: string, teamId: number) {
  return request<SquadResponse>(baseUrl, `/squad?team_id=${teamId}`);
}

export function postRecommend(baseUrl: string, body: SquadRequestBody) {
  return request<RecommendResponse>(baseUrl, "/recommend", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function postChips(baseUrl: string, body: SquadRequestBody) {
  return request<ChipsResponse>(baseUrl, "/chips", {
    method: "POST",
    body: JSON.stringify(body),
  });
}
