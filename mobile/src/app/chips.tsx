import { useState } from "react";
import { View, Text, StyleSheet, ScrollView } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import Ionicons from "@expo/vector-icons/Ionicons";

import { getApiBaseUrl, getTeamId } from "@/lib/config";
import { postChips, ChipsResponse, ChipScore, ApiError } from "@/lib/api";
import { colors, spacing, type } from "@/lib/theme";
import { Card, PrimaryButton, ScoreBadge } from "@/components/ui";

export default function ChipsScreen() {
  const [result, setResult] = useState<ChipsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function fetchChips() {
    const baseUrl = await getApiBaseUrl();
    const id = await getTeamId();
    if (!id) {
      setError("Set your Team ID in the Settings tab first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      setResult(await postChips(baseUrl, { team_id: Number(id), horizon: 3 }));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <SafeAreaView style={styles.safeArea} edges={["bottom"]}>
      <ScrollView style={styles.container} contentContainerStyle={styles.content}>
        <Text style={styles.intro}>
          Values are for your CURRENT squad, before any transfer is applied — don&apos;t act on
          one week&apos;s reading alone.
        </Text>

        <PrimaryButton label="Get Chip Values" onPress={fetchChips} loading={loading} />

        {error && (
          <View style={styles.errorBox}>
            <Ionicons name="alert-circle" size={18} color={colors.danger} />
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        {result && (
          <View style={styles.resultBlock}>
            <ChipCard name="Bench Boost" icon="people" byWeek={result.bench_boost.by_week} score={result.bench_boost.score} startGw={result.gw} />
            <ChipCard name="Triple Captain" icon="star" byWeek={result.triple_captain.by_week} score={result.triple_captain.score} startGw={result.gw} />
            <SingleValueChipCard name="Wildcard" icon="refresh-circle" gain={result.wildcard.gain} score={result.wildcard.score} />
            <SingleValueChipCard name="Free Hit" icon="flash" gain={result.free_hit.gain} score={result.free_hit.score} />
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function ChipCard({
  name, icon, byWeek, score, startGw,
}: {
  name: string;
  icon: keyof typeof Ionicons.glyphMap;
  byWeek: Record<string, number>;
  score: ChipScore;
  startGw: number;
}) {
  const weeks = Object.entries(byWeek).sort(([a], [b]) => Number(a) - Number(b));
  const bestOffset = weeks.reduce((best, [w, v]) => (v > byWeek[best] ? w : best), weeks[0]?.[0] ?? "0");

  return (
    <Card>
      <View style={styles.cardHeader}>
        <View style={styles.cardTitleRow}>
          <Ionicons name={icon} size={18} color={colors.primary} />
          <Text style={styles.cardTitle}>{name}</Text>
        </View>
        <ScoreBadge score={score.score} />
      </View>
      {weeks.map(([offset, value]) => (
        <View key={offset} style={styles.weekRow}>
          <Text style={styles.weekLabel}>
            GW{startGw + Number(offset)}{offset === bestOffset ? "  ⭐ best" : ""}
          </Text>
          <Text style={styles.weekValue}>+{value.toFixed(2)} pts</Text>
        </View>
      ))}
      <Text style={styles.verdict}>{score.verdict}</Text>
    </Card>
  );
}

function SingleValueChipCard({
  name, icon, gain, score,
}: {
  name: string; icon: keyof typeof Ionicons.glyphMap; gain: number; score: ChipScore;
}) {
  return (
    <Card>
      <View style={styles.cardHeader}>
        <View style={styles.cardTitleRow}>
          <Ionicons name={icon} size={18} color={colors.primary} />
          <Text style={styles.cardTitle}>{name}</Text>
        </View>
        <ScoreBadge score={score.score} />
      </View>
      <Text style={styles.weekValue}>+{gain.toFixed(2)} pts</Text>
      <Text style={styles.verdict}>{score.verdict}</Text>
    </Card>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: colors.background },
  container: { flex: 1 },
  content: { padding: spacing.md, paddingBottom: spacing.xl },
  intro: { ...type.caption, marginBottom: spacing.md, lineHeight: 16 },
  errorBox: {
    flexDirection: "row", alignItems: "center", gap: spacing.sm,
    backgroundColor: "#fdecef", borderRadius: 10, padding: spacing.sm, marginTop: spacing.md,
  },
  errorText: { color: colors.danger, flex: 1, fontSize: 13 },
  resultBlock: { marginTop: spacing.md },
  cardHeader: {
    flexDirection: "row", justifyContent: "space-between",
    alignItems: "center", marginBottom: spacing.sm,
  },
  cardTitleRow: { flexDirection: "row", alignItems: "center", gap: 6 },
  cardTitle: { ...type.subtitle },
  weekRow: { flexDirection: "row", justifyContent: "space-between", paddingVertical: 3 },
  weekLabel: { fontSize: 13, color: colors.textPrimary },
  weekValue: { fontSize: 14, fontWeight: "700", color: colors.primary },
  verdict: { ...type.caption, marginTop: spacing.sm, lineHeight: 16 },
});
