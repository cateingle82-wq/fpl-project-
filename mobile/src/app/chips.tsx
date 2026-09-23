import { useState } from "react";
import { View, Text, Pressable, StyleSheet, ScrollView, ActivityIndicator } from "react-native";

import { getApiBaseUrl, getTeamId } from "@/lib/config";
import { postChips, ChipsResponse, ChipScore, ApiError } from "@/lib/api";

export default function ChipsScreen() {
  const [result, setResult] = useState<ChipsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function fetchChips() {
    const baseUrl = await getApiBaseUrl();
    const id = await getTeamId();
    if (!id) {
      setError("Set your Team ID in Settings first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await postChips(baseUrl, { team_id: Number(id), horizon: 3 });
      setResult(data);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.intro}>
        Values are for your CURRENT squad, before any transfer is applied — don&apos;t act on
        one week&apos;s reading alone.
      </Text>

      <Pressable
        style={[styles.button, loading && styles.buttonDisabled]}
        onPress={fetchChips}
        disabled={loading}
      >
        {loading
          ? <ActivityIndicator color="white" />
          : <Text style={styles.buttonText}>Get Chip Values</Text>}
      </Pressable>

      {error && <Text style={styles.error}>{error}</Text>}

      {result && (
        <View style={styles.resultBlock}>
          <Text style={styles.gwTitle}>GW{result.gw} — {result.horizon}-week horizon</Text>

          <ChipCard
            name="Bench Boost"
            byWeek={result.bench_boost.by_week}
            score={result.bench_boost.score}
            startGw={result.gw}
          />
          <ChipCard
            name="Triple Captain"
            byWeek={result.triple_captain.by_week}
            score={result.triple_captain.score}
            startGw={result.gw}
          />
          <SingleValueChipCard name="Wildcard" gain={result.wildcard.gain} score={result.wildcard.score} />
          <SingleValueChipCard name="Free Hit" gain={result.free_hit.gain} score={result.free_hit.score} />
        </View>
      )}
    </ScrollView>
  );
}

function ChipCard({
  name, byWeek, score, startGw,
}: {
  name: string;
  byWeek: Record<string, number>;
  score: ChipScore;
  startGw: number;
}) {
  const weeks = Object.entries(byWeek).sort(([a], [b]) => Number(a) - Number(b));
  const bestOffset = weeks.reduce((best, [w, v]) => (v > byWeek[best] ? w : best), weeks[0]?.[0] ?? "0");

  return (
    <View style={styles.card}>
      <View style={styles.cardHeader}>
        <Text style={styles.cardTitle}>{name}</Text>
        <Text style={styles.cardScore}>
          {score.score !== null ? `${score.score}/10` : "n/a"}
        </Text>
      </View>
      {weeks.map(([offset, value]) => (
        <View key={offset} style={styles.weekRow}>
          <Text style={styles.weekLabel}>
            GW{startGw + Number(offset)}{offset === bestOffset ? " ★" : ""}
          </Text>
          <Text style={styles.weekValue}>+{value.toFixed(2)} pts</Text>
        </View>
      ))}
      <Text style={styles.verdict}>{score.verdict}</Text>
    </View>
  );
}

function SingleValueChipCard({ name, gain, score }: { name: string; gain: number; score: ChipScore }) {
  return (
    <View style={styles.card}>
      <View style={styles.cardHeader}>
        <Text style={styles.cardTitle}>{name}</Text>
        <Text style={styles.cardScore}>
          {score.score !== null ? `${score.score}/10` : "n/a"}
        </Text>
      </View>
      <Text style={styles.weekValue}>+{gain.toFixed(2)} pts</Text>
      <Text style={styles.verdict}>{score.verdict}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "white" },
  content: { padding: 16, paddingBottom: 48 },
  intro: { color: "#666", fontSize: 13, marginBottom: 16 },
  button: {
    backgroundColor: "#37003c", borderRadius: 8, padding: 14,
    alignItems: "center", justifyContent: "center", minHeight: 48,
  },
  buttonDisabled: { opacity: 0.6 },
  buttonText: { color: "white", fontWeight: "600", fontSize: 16 },
  error: { color: "#c00", marginTop: 12 },
  resultBlock: { marginTop: 20 },
  gwTitle: { fontSize: 18, fontWeight: "700", marginBottom: 12 },
  card: {
    borderWidth: 1, borderColor: "#eee", borderRadius: 10,
    padding: 14, marginBottom: 12,
  },
  cardHeader: {
    flexDirection: "row", justifyContent: "space-between",
    alignItems: "center", marginBottom: 6,
  },
  cardTitle: { fontWeight: "700", fontSize: 15 },
  cardScore: { fontWeight: "700", fontSize: 15, color: "#37003c" },
  weekRow: { flexDirection: "row", justifyContent: "space-between", paddingVertical: 2 },
  weekLabel: { fontSize: 13, color: "#444" },
  weekValue: { fontSize: 13, fontWeight: "600" },
  verdict: { color: "#666", fontSize: 12, marginTop: 6 },
});
