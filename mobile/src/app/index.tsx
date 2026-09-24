import { useCallback, useState } from "react";
import { View, Text, StyleSheet, ScrollView, RefreshControl, Pressable } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useFocusEffect } from "expo-router";
import Ionicons from "@expo/vector-icons/Ionicons";

import { getApiBaseUrl, getTeamId, getHorizon } from "@/lib/config";
import { postRecommend, RecommendResponse, PlanWeek, ApiError } from "@/lib/api";
import { colors, spacing, type } from "@/lib/theme";
import { Card, SectionTitle, PrimaryButton, Metric, PlayerRow } from "@/components/ui";

export default function HomeScreen() {
  const [teamId, setTeamId] = useState("");
  const [result, setResult] = useState<RecommendResponse | null>(null);
  const [selectedWeek, setSelectedWeek] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useFocusEffect(
    useCallback(() => {
      (async () => setTeamId(await getTeamId()))();
    }, [])
  );

  async function fetchRecommendation() {
    const baseUrl = await getApiBaseUrl();
    const id = await getTeamId();
    const horizon = await getHorizon();
    if (!id) {
      setError("Set your Team ID in the Settings tab first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await postRecommend(baseUrl, { team_id: Number(id), horizon });
      setResult(data);
      setSelectedWeek(0);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <SafeAreaView style={styles.safeArea} edges={["bottom"]}>
      <ScrollView
        style={styles.container}
        contentContainerStyle={styles.content}
        refreshControl={<RefreshControl refreshing={loading} onRefresh={fetchRecommendation} />}
      >
        {!teamId && (
          <View style={styles.emptyState}>
            <Ionicons name="person-circle-outline" size={48} color={colors.textSecondary} />
            <Text style={styles.emptyStateText}>Set your Team ID in the Settings tab to get started.</Text>
          </View>
        )}

        <PrimaryButton label="Get Recommendation" onPress={fetchRecommendation} loading={loading} />

        {error && (
          <View style={styles.errorBox}>
            <Ionicons name="alert-circle" size={18} color={colors.danger} />
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        {result && (
          <ResultView result={result} selectedWeek={selectedWeek} onSelectWeek={setSelectedWeek} />
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function ResultView({
  result, selectedWeek, onSelectWeek,
}: {
  result: RecommendResponse;
  selectedWeek: number;
  onSelectWeek: (w: number) => void;
}) {
  const { transfers } = result;
  const week: PlanWeek = result.plan[selectedWeek];

  return (
    <View>
      <Card style={styles.headerCard}>
        <Text style={styles.gwTitle}>GW{result.gw}</Text>
        <Text style={styles.gwSubtitle}>{result.horizon}-week plan</Text>
        <View style={styles.metricsRow}>
          <Metric light label="Objective" value={result.objective.toFixed(1)} />
          <Metric light label="Avg / GW" value={result.average_per_gw.toFixed(1)} />
          <Metric light label="Cost" value={`£${result.squad_cost.toFixed(1)}m`} />
        </View>
      </Card>

      <SectionTitle>This Week&apos;s Transfer</SectionTitle>
      <Card>
        {transfers.out.length === 0 ? (
          <Text style={type.body}>No transfer — roll it.</Text>
        ) : (
          <>
            <View style={styles.transferBanner}>
              <Text style={styles.transferBannerText}>
                {transfers.out.length} transfer{transfers.out.length > 1 ? "s" : ""}
                {transfers.hits > 0
                  ? ` · ${transfers.hits} hit${transfers.hits > 1 ? "s" : ""} (-${transfers.hit_cost_paid.toFixed(0)} pts)`
                  : " · free"}
              </Text>
            </View>
            <View style={styles.transferCols}>
              <View style={styles.transferCol}>
                <Text style={styles.transferHeader}>OUT</Text>
                {transfers.out.map((p) => <PlayerRow key={p.id} player={p} subtle />)}
              </View>
              <View style={styles.transferCol}>
                <Text style={styles.transferHeader}>IN</Text>
                {transfers.in.map((p) => <PlayerRow key={p.id} player={p} />)}
              </View>
            </View>
          </>
        )}
      </Card>

      <SectionTitle>Weekly Plan</SectionTitle>
      <Text style={styles.planCaption}>
        Only THIS week&apos;s transfer above is real — everything here beyond it is the
        model&apos;s own forward-looking plan, re-solved fresh every week.
      </Text>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.weekScroll}>
        {result.plan.map((w) => (
          <Pressable
            key={w.week_offset}
            style={[styles.weekPill, selectedWeek === w.week_offset && styles.weekPillActive]}
            onPress={() => onSelectWeek(w.week_offset)}
          >
            <Text style={[styles.weekPillText, selectedWeek === w.week_offset && styles.weekPillTextActive]}>
              GW{w.gw}
            </Text>
          </Pressable>
        ))}
      </ScrollView>

      {week.transferred_in !== undefined && (
        <Card style={styles.planChangeCard}>
          {week.transferred_in.length === 0 && week.transferred_out!.length === 0 ? (
            <Text style={type.caption}>No change from the previous week.</Text>
          ) : (
            <>
              <Text style={styles.planChangeTitle}>
                Planned transfer: {week.hits! > 0
                  ? `${week.transferred_out!.length} in, ${week.hits} hit(s)`
                  : "free"}
              </Text>
              {week.transferred_out!.map((p) => (
                <Text key={p.id} style={type.caption}>OUT {p.name}</Text>
              ))}
              {week.transferred_in.map((p) => (
                <Text key={p.id} style={type.caption}>IN {p.name}</Text>
              ))}
            </>
          )}
        </Card>
      )}

      <SectionTitle>Captain</SectionTitle>
      <Card><PlayerRow player={week.captain} /></Card>

      <SectionTitle>Starting XI</SectionTitle>
      <Card>{week.xi.map((p) => <PlayerRow key={p.id} player={p} />)}</Card>

      <SectionTitle>Bench</SectionTitle>
      <Card>{week.bench.map((p) => <PlayerRow key={p.id} player={p} subtle />)}</Card>
    </View>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: colors.background },
  container: { flex: 1 },
  content: { padding: spacing.md, paddingBottom: spacing.xl },
  emptyState: { alignItems: "center", paddingVertical: spacing.lg, gap: spacing.sm },
  emptyStateText: { ...type.body, color: colors.textSecondary, textAlign: "center" },
  errorBox: {
    flexDirection: "row", alignItems: "center", gap: spacing.sm,
    backgroundColor: "#fdecef", borderRadius: 10, padding: spacing.sm, marginTop: spacing.md,
  },
  errorText: { color: colors.danger, flex: 1, fontSize: 13 },
  headerCard: { marginTop: spacing.md, backgroundColor: colors.primary },
  gwTitle: { fontSize: 26, fontWeight: "800", color: colors.textOnPrimary },
  gwSubtitle: { color: "#d9c2db", marginBottom: spacing.md },
  metricsRow: { flexDirection: "row", justifyContent: "space-between" },
  transferBanner: {
    backgroundColor: colors.background, borderRadius: 8,
    paddingVertical: 6, paddingHorizontal: 10, marginBottom: spacing.sm, alignSelf: "flex-start",
  },
  transferBannerText: { fontWeight: "700", fontSize: 12, color: colors.primary },
  transferCols: { flexDirection: "row", gap: spacing.md },
  transferCol: { flex: 1 },
  transferHeader: { ...type.metricLabel, marginBottom: spacing.xs },
  planCaption: { ...type.caption, marginBottom: spacing.sm, lineHeight: 16 },
  weekScroll: { marginBottom: spacing.sm },
  weekPill: {
    paddingHorizontal: 14, paddingVertical: 8, borderRadius: 999,
    backgroundColor: colors.surface, borderWidth: 1, borderColor: colors.border,
    marginRight: spacing.sm,
  },
  weekPillActive: { backgroundColor: colors.primary, borderColor: colors.primary },
  weekPillText: { fontWeight: "600", fontSize: 13, color: colors.textPrimary },
  weekPillTextActive: { color: colors.textOnPrimary },
  planChangeCard: { backgroundColor: "#f0e6f1" },
  planChangeTitle: { fontWeight: "700", fontSize: 13, color: colors.primary, marginBottom: 4 },
});
