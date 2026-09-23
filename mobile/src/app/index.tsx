import { useCallback, useState } from "react";
import {
  View, Text, Pressable, StyleSheet, ScrollView,
  ActivityIndicator, RefreshControl,
} from "react-native";
import { router, useFocusEffect } from "expo-router";

import { getApiBaseUrl, getTeamId } from "@/lib/config";
import { postRecommend, RecommendResponse, PlayerSummary, ApiError } from "@/lib/api";

export default function HomeScreen() {
  const [apiBaseUrl, setApiBaseUrl] = useState("");
  const [teamId, setTeamId] = useState("");
  const [result, setResult] = useState<RecommendResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Re-read settings every time this screen regains focus (e.g. coming
  // back from Settings) — reading from local component state that was
  // only set on mount would show stale values after an edit.
  useFocusEffect(
    useCallback(() => {
      (async () => {
        setApiBaseUrl(await getApiBaseUrl());
        setTeamId(await getTeamId());
      })();
    }, [])
  );

  async function fetchRecommendation() {
    // Read fresh from storage rather than trusting component state for
    // the actual network call — belt-and-braces against the state above
    // somehow lagging a Settings edit.
    const baseUrl = await getApiBaseUrl();
    const id = await getTeamId();
    if (!id) {
      setError("Set your Team ID in Settings first.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const data = await postRecommend(baseUrl, { team_id: Number(id), horizon: 3 });
      setResult(data);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      refreshControl={<RefreshControl refreshing={loading} onRefresh={fetchRecommendation} />}
    >
      <View style={styles.statusRow}>
        <Text style={styles.statusText}>
          {teamId ? `Team ${teamId}` : "No team set"} · {apiBaseUrl || "no API URL set"}
        </Text>
        <Pressable onPress={() => router.push("/settings")}>
          <Text style={styles.settingsLink}>Settings</Text>
        </Pressable>
      </View>

      <Pressable
        style={[styles.button, loading && styles.buttonDisabled]}
        onPress={fetchRecommendation}
        disabled={loading}
      >
        {loading
          ? <ActivityIndicator color="white" />
          : <Text style={styles.buttonText}>Get Recommendation</Text>}
      </Pressable>

      {error && <Text style={styles.error}>{error}</Text>}

      {result && <ResultView result={result} />}
    </ScrollView>
  );
}

function ResultView({ result }: { result: RecommendResponse }) {
  const { transfers } = result;
  return (
    <View style={styles.resultBlock}>
      <Text style={styles.gwTitle}>GW{result.gw} — {result.horizon}-week plan</Text>
      <View style={styles.metricsRow}>
        <Metric label="Objective" value={result.objective.toFixed(1)} />
        <Metric label="Avg / GW" value={result.average_per_gw.toFixed(1)} />
        <Metric label="Squad cost" value={`£${result.squad_cost.toFixed(1)}m`} />
      </View>

      <Text style={styles.sectionTitle}>Transfers</Text>
      {transfers.out.length === 0 ? (
        <Text style={styles.bodyText}>No transfer — roll it.</Text>
      ) : (
        <>
          <Text style={styles.bodyText}>
            {transfers.out.length} transfer(s), {transfers.hits} hit(s)
            {transfers.hits > 0 ? ` (-${transfers.hit_cost_paid.toFixed(0)} pts)` : ""}
          </Text>
          <View style={styles.transferCols}>
            <View style={styles.transferCol}>
              <Text style={styles.transferHeader}>OUT</Text>
              {transfers.out.map((p) => <PlayerRow key={p.id} player={p} />)}
            </View>
            <View style={styles.transferCol}>
              <Text style={styles.transferHeader}>IN</Text>
              {transfers.in.map((p) => <PlayerRow key={p.id} player={p} />)}
            </View>
          </View>
        </>
      )}

      <Text style={styles.sectionTitle}>Captain</Text>
      <PlayerRow player={result.captain} />

      <Text style={styles.sectionTitle}>Starting XI</Text>
      {result.xi.map((p) => <PlayerRow key={p.id} player={p} />)}

      <Text style={styles.sectionTitle}>Bench</Text>
      {result.bench.map((p) => <PlayerRow key={p.id} player={p} />)}
    </View>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.metric}>
      <Text style={styles.metricValue}>{value}</Text>
      <Text style={styles.metricLabel}>{label}</Text>
    </View>
  );
}

function PlayerRow({ player }: { player: PlayerSummary }) {
  return (
    <View style={styles.playerRow}>
      <Text style={styles.playerName}>
        {player.name}{player.captain ? " (C)" : ""}
      </Text>
      <Text style={styles.playerMeta}>{player.pos} · £{player.price.toFixed(1)}m</Text>
      <Text style={styles.playerXpts}>{player.xpts.toFixed(1)}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "white" },
  content: { padding: 16, paddingBottom: 48 },
  statusRow: {
    flexDirection: "row", justifyContent: "space-between",
    alignItems: "center", marginBottom: 16,
  },
  statusText: { color: "#666", fontSize: 13 },
  settingsLink: { color: "#37003c", fontWeight: "600" },
  button: {
    backgroundColor: "#37003c", borderRadius: 8, padding: 14,
    alignItems: "center", justifyContent: "center", minHeight: 48,
  },
  buttonDisabled: { opacity: 0.6 },
  buttonText: { color: "white", fontWeight: "600", fontSize: 16 },
  error: { color: "#c00", marginTop: 12 },
  resultBlock: { marginTop: 20 },
  gwTitle: { fontSize: 18, fontWeight: "700", marginBottom: 12 },
  metricsRow: { flexDirection: "row", justifyContent: "space-between", marginBottom: 16 },
  metric: { alignItems: "center" },
  metricValue: { fontSize: 20, fontWeight: "700" },
  metricLabel: { color: "#666", fontSize: 12 },
  sectionTitle: { fontWeight: "700", fontSize: 15, marginTop: 16, marginBottom: 6 },
  bodyText: { fontSize: 14 },
  transferCols: { flexDirection: "row", gap: 16 },
  transferCol: { flex: 1 },
  transferHeader: { fontWeight: "600", color: "#666", marginBottom: 4 },
  playerRow: {
    flexDirection: "row", justifyContent: "space-between",
    paddingVertical: 4, borderBottomWidth: StyleSheet.hairlineWidth, borderColor: "#eee",
  },
  playerName: { flex: 2, fontSize: 14 },
  playerMeta: { flex: 1, fontSize: 12, color: "#666" },
  playerXpts: { width: 40, textAlign: "right", fontSize: 14, fontWeight: "600" },
});
