/**
 * A literal formation layout (attack at top, keeper at bottom) instead
 * of a plain player list — the single biggest "does this look like a
 * real product" change available for the effort, same idea as the
 * Streamlit dashboard's render_pitch. Pure Views/borders, no SVG or
 * gradient library needed (none installed, and none of this needs one).
 */
import { View, Text, Pressable, StyleSheet, Alert, ScrollView } from "react-native";
import Ionicons from "@expo/vector-icons/Ionicons";

import { colors, spacing, radius, shadow } from "@/lib/theme";
import type { PlayerSummary } from "@/lib/api";
import { RISK_STYLE } from "@/components/ui";

const PITCH_GREEN = "#1e7a45";
const PITCH_LINE = "rgba(255,255,255,0.35)";
const ROW_ORDER = ["FWD", "MID", "DEF", "GKP"] as const;

export function PitchView({ xi, bench }: { xi: PlayerSummary[]; bench: PlayerSummary[] }) {
  const byPos: Record<string, PlayerSummary[]> = {};
  for (const p of xi) {
    (byPos[p.pos] ??= []).push(p);
  }

  return (
    <View>
      <View style={styles.pitch}>
        <View style={styles.centerCircle} />
        <View style={styles.halfwayLine} />
        {ROW_ORDER.map((pos) => {
          const players = byPos[pos];
          if (!players || players.length === 0) return null;
          return (
            <View key={pos} style={styles.row}>
              {players
                .slice()
                .sort((a, b) => b.xpts - a.xpts)
                .map((p) => <PitchCard key={p.id} player={p} />)}
            </View>
          );
        })}
      </View>

      {bench.length > 0 && (
        <View style={styles.benchSection}>
          <Text style={styles.benchLabel}>BENCH</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            <View style={styles.benchRow}>
              {bench.map((p) => <PitchCard key={p.id} player={p} bench />)}
            </View>
          </ScrollView>
        </View>
      )}
    </View>
  );
}

function PitchCard({ player, bench }: { player: PlayerSummary; bench?: boolean }) {
  const risk = player.risk.level !== "ok" ? RISK_STYLE[player.risk.level] : null;

  return (
    <View style={[styles.card, bench && styles.benchCard]}>
      {player.captain && (
        <View style={styles.captainBadge}>
          <Text style={styles.captainBadgeText}>C</Text>
        </View>
      )}
      {risk && (
        <Pressable
          style={styles.riskBadge}
          hitSlop={6}
          onPress={() => Alert.alert(player.name, player.risk.detail ?? "Flagged")}
        >
          <Ionicons name={risk.icon} size={13} color={risk.color} />
        </Pressable>
      )}
      <Text style={styles.cardName} numberOfLines={1}>{player.name}</Text>
      <Text style={styles.cardMeta} numberOfLines={1}>£{player.price.toFixed(1)}m</Text>
      <Text style={styles.cardXpts}>{player.xpts.toFixed(1)}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  pitch: {
    backgroundColor: PITCH_GREEN,
    borderRadius: radius.lg,
    paddingVertical: spacing.lg,
    paddingHorizontal: spacing.sm,
    marginBottom: spacing.md,
    overflow: "hidden",
    gap: spacing.md,
  },
  halfwayLine: {
    position: "absolute",
    left: 0, right: 0, top: "50%",
    height: 1, backgroundColor: PITCH_LINE,
  },
  centerCircle: {
    position: "absolute",
    top: "50%", left: "50%",
    width: 70, height: 70, borderRadius: 35,
    marginLeft: -35, marginTop: -35,
    borderWidth: 1, borderColor: PITCH_LINE,
  },
  row: {
    flexDirection: "row",
    justifyContent: "space-evenly",
    flexWrap: "wrap",
  },
  card: {
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    paddingVertical: 6,
    paddingHorizontal: 4,
    width: 68,
    alignItems: "center",
    ...shadow,
  },
  benchCard: {
    backgroundColor: "#f0f0f2",
    marginRight: spacing.sm,
  },
  cardName: { fontSize: 11, fontWeight: "700", color: colors.textPrimary, maxWidth: 62 },
  cardMeta: { fontSize: 9, color: colors.textSecondary },
  cardXpts: { fontSize: 12, fontWeight: "800", color: colors.primary, marginTop: 2 },
  captainBadge: {
    position: "absolute", top: -6, right: -6,
    width: 18, height: 18, borderRadius: 9,
    backgroundColor: "#ffd166", alignItems: "center", justifyContent: "center",
    borderWidth: 1, borderColor: colors.surface, zIndex: 2,
  },
  captainBadgeText: { fontSize: 10, fontWeight: "800", color: colors.primary },
  riskBadge: {
    position: "absolute", top: -6, left: -6,
    width: 18, height: 18, borderRadius: 9,
    backgroundColor: colors.surface, alignItems: "center", justifyContent: "center",
    zIndex: 2,
  },
  benchSection: { marginBottom: spacing.md },
  benchLabel: {
    fontSize: 11, fontWeight: "700", color: colors.textSecondary,
    letterSpacing: 0.5, marginBottom: spacing.xs,
  },
  benchRow: { flexDirection: "row" },
});
