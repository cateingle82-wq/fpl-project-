/**
 * Small shared building blocks used across every screen — kept in one
 * file since none of these are complex enough to need their own, and
 * having them in one place makes it obvious when a screen is about to
 * duplicate styling instead of reusing it.
 */
import { View, Text, Pressable, StyleSheet, ActivityIndicator } from "react-native";
import { colors, spacing, radius, type, shadow, positionColors } from "@/lib/theme";
import type { PlayerSummary } from "@/lib/api";

export function Card({ children, style }: { children: React.ReactNode; style?: object }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

export function SectionTitle({ children }: { children: React.ReactNode }) {
  return <Text style={styles.sectionTitle}>{children}</Text>;
}

export function PrimaryButton({
  label, onPress, loading, disabled,
}: {
  label: string; onPress: () => void; loading?: boolean; disabled?: boolean;
}) {
  return (
    <Pressable
      style={({ pressed }) => [
        styles.button,
        (disabled || loading) && styles.buttonDisabled,
        pressed && !disabled && !loading && styles.buttonPressed,
      ]}
      onPress={onPress}
      disabled={disabled || loading}
    >
      {loading ? <ActivityIndicator color={colors.textOnPrimary} /> : <Text style={styles.buttonText}>{label}</Text>}
    </Pressable>
  );
}

export function PositionBadge({ pos }: { pos: string }) {
  const bg = positionColors[pos] ?? colors.textSecondary;
  return (
    <View style={[styles.badge, { backgroundColor: bg }]}>
      <Text style={styles.badgeText}>{pos}</Text>
    </View>
  );
}

export function Metric({
  label, value, accent, light,
}: {
  label: string; value: string; accent?: boolean; light?: boolean;
}) {
  return (
    <View style={styles.metric}>
      <Text style={[
        type.metricValue,
        light && { color: colors.textOnPrimary },
        accent && { color: colors.danger },
      ]}>
        {value}
      </Text>
      <Text style={[type.metricLabel, light && { color: "#d9c2db" }]}>{label}</Text>
    </View>
  );
}

export function ScoreBadge({ score }: { score: number | null }) {
  if (score === null) {
    return (
      <View style={[styles.scorePill, { backgroundColor: colors.border }]}>
        <Text style={[styles.scorePillText, { color: colors.textSecondary }]}>n/a</Text>
      </View>
    );
  }
  const bg = score >= 7 ? colors.accent : score >= 4 ? "#ffd166" : colors.danger;
  const fg = score >= 7 ? colors.primary : colors.textOnPrimary;
  return (
    <View style={[styles.scorePill, { backgroundColor: bg }]}>
      <Text style={[styles.scorePillText, { color: fg }]}>{score}/10</Text>
    </View>
  );
}

export function PlayerRow({ player, subtle }: { player: PlayerSummary; subtle?: boolean }) {
  return (
    <View style={[styles.playerRow, subtle && { opacity: 0.6 }]}>
      <PositionBadge pos={player.pos} />
      <View style={styles.playerInfo}>
        <Text style={styles.playerName} numberOfLines={1}>
          {player.name}{player.captain ? " (C)" : ""}
        </Text>
        <Text style={type.caption}>{player.team} · £{player.price.toFixed(1)}m</Text>
      </View>
      <Text style={styles.playerXpts}>{player.xpts.toFixed(1)}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderRadius: radius.lg,
    padding: spacing.md,
    marginBottom: spacing.md,
    ...shadow,
  },
  sectionTitle: {
    ...type.subtitle,
    marginTop: spacing.lg,
    marginBottom: spacing.sm,
  },
  button: {
    backgroundColor: colors.primary,
    borderRadius: radius.md,
    paddingVertical: 14,
    alignItems: "center",
    justifyContent: "center",
    minHeight: 50,
  },
  buttonPressed: { backgroundColor: colors.primaryLight },
  buttonDisabled: { opacity: 0.5 },
  buttonText: { color: colors.textOnPrimary, fontWeight: "700", fontSize: 16 },
  badge: {
    borderRadius: radius.sm,
    paddingHorizontal: 8,
    paddingVertical: 3,
    minWidth: 38,
    alignItems: "center",
  },
  badgeText: { color: "white", fontWeight: "700", fontSize: 11 },
  metric: { alignItems: "center", flex: 1 },
  scorePill: { borderRadius: radius.pill, paddingHorizontal: 10, paddingVertical: 4 },
  scorePillText: { fontWeight: "700", fontSize: 13 },
  playerRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    gap: spacing.sm,
  },
  playerInfo: { flex: 1, minWidth: 0 },
  playerName: { fontSize: 14, fontWeight: "600", color: colors.textPrimary },
  playerXpts: { fontWeight: "700", fontSize: 14, color: colors.primary, width: 40, textAlign: "right" },
});
