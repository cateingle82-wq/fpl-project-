import { useEffect, useState } from "react";
import { View, Text, TextInput, StyleSheet, Alert, ScrollView, Switch } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import {
  DEFAULT_API_BASE_URL,
  DEFAULT_HORIZON,
  DEFAULT_REMINDER_HOURS_BEFORE,
  getApiBaseUrl,
  setApiBaseUrl,
  getTeamId,
  setTeamId,
  getHorizon,
  setHorizon,
  getApiKey,
  setApiKey,
  getReminderEnabled,
  setReminderEnabled,
  getReminderHoursBefore,
  setReminderHoursBefore,
} from "@/lib/config";
import {
  requestNotificationPermission,
  scheduleDeadlineReminder,
  cancelDeadlineReminder,
  ensureAndroidChannel,
} from "@/lib/notifications";
import { colors, spacing, radius, type } from "@/lib/theme";
import { Card, PrimaryButton } from "@/components/ui";

export default function SettingsScreen() {
  const [apiBaseUrl, setApiBaseUrlInput] = useState(DEFAULT_API_BASE_URL);
  const [teamId, setTeamIdInput] = useState("");
  const [horizon, setHorizonInput] = useState(String(DEFAULT_HORIZON));
  const [apiKey, setApiKeyInput] = useState("");
  const [reminderEnabled, setReminderEnabledInput] = useState(false);
  const [reminderHours, setReminderHoursInput] = useState(String(DEFAULT_REMINDER_HOURS_BEFORE));
  const [loaded, setLoaded] = useState(false);
  const [savedFlash, setSavedFlash] = useState(false);

  useEffect(() => {
    (async () => {
      setApiBaseUrlInput(await getApiBaseUrl());
      setTeamIdInput(await getTeamId());
      setHorizonInput(String(await getHorizon()));
      setApiKeyInput(await getApiKey());
      setReminderEnabledInput(await getReminderEnabled());
      setReminderHoursInput(String(await getReminderHoursBefore()));
      await ensureAndroidChannel();
      setLoaded(true);
    })();
  }, []);

  async function save() {
    if (teamId.trim() && Number.isNaN(Number(teamId.trim()))) {
      Alert.alert("Invalid Team ID", "Team ID must be a number — it's the number in your FPL team's URL.");
      return;
    }
    const horizonNum = Number(horizon.trim());
    if (!Number.isInteger(horizonNum) || horizonNum < 1 || horizonNum > 8) {
      Alert.alert("Invalid Horizon", "Horizon must be a whole number between 1 and 8 weeks.");
      return;
    }
    const reminderHoursNum = Number(reminderHours.trim());
    if (!Number.isFinite(reminderHoursNum) || reminderHoursNum <= 0 || reminderHoursNum > 48) {
      Alert.alert("Invalid Reminder Time", "Hours before deadline must be between 1 and 48.");
      return;
    }
    await setApiBaseUrl(apiBaseUrl);
    await setTeamId(teamId);
    await setHorizon(horizonNum);
    await setApiKey(apiKey);
    await setReminderEnabled(reminderEnabled);
    await setReminderHoursBefore(reminderHoursNum);

    if (reminderEnabled) {
      const granted = await requestNotificationPermission();
      if (!granted) {
        Alert.alert("Notifications disabled", "Enable notifications for this app in your "
          + "phone's system settings for the deadline reminder to actually alert you.");
      } else {
        try {
          const scheduledFor = await scheduleDeadlineReminder();
          if (!scheduledFor) {
            Alert.alert("Nothing to schedule yet", "The next deadline is already less than "
              + `${reminderHoursNum}h away, or couldn't be fetched — try again closer to a `
              + "gameweek, or check your API base URL/key.");
          }
        } catch {
          Alert.alert("Couldn't schedule reminder", "Check your API base URL and API Key are "
            + "correct, then try saving again.");
        }
      }
    } else {
      await cancelDeadlineReminder();
    }

    setSavedFlash(true);
    setTimeout(() => setSavedFlash(false), 1800);
  }

  if (!loaded) return null;

  return (
    <SafeAreaView style={styles.safeArea} edges={["bottom"]}>
      <ScrollView style={styles.container} contentContainerStyle={styles.content}>
        <Card>
          <Text style={styles.label}>API base URL</Text>
          <TextInput
            style={styles.input}
            value={apiBaseUrl}
            onChangeText={setApiBaseUrlInput}
            autoCapitalize="none"
            autoCorrect={false}
            placeholder={DEFAULT_API_BASE_URL}
            placeholderTextColor={colors.textSecondary}
          />
          <Text style={styles.hint}>
            localhost only works from the iOS Simulator. On a physical phone, use your dev
            machine&apos;s LAN IP (e.g. http://192.168.1.42:8000) — the server must be started
            with `uvicorn api:app --host 0.0.0.0`. On the Android emulator, use
            http://10.0.2.2:8000.
          </Text>
        </Card>

        <Card>
          <Text style={styles.label}>FPL Team ID</Text>
          <TextInput
            style={styles.input}
            value={teamId}
            onChangeText={setTeamIdInput}
            keyboardType="number-pad"
            placeholder="e.g. 7362936"
            placeholderTextColor={colors.textSecondary}
          />
          <Text style={styles.hint}>The number in your FPL team&apos;s URL.</Text>
        </Card>

        <Card>
          <Text style={styles.label}>Planning horizon (weeks)</Text>
          <TextInput
            style={styles.input}
            value={horizon}
            onChangeText={setHorizonInput}
            keyboardType="number-pad"
            placeholder={String(DEFAULT_HORIZON)}
            placeholderTextColor={colors.textSecondary}
          />
          <Text style={styles.hint}>
            How many gameweeks ahead the optimiser plans over (1-8). A longer horizon lets it
            reason about banking a transfer now for a bigger swap later — see the Weekly Plan
            scroll on the Squad tab.
          </Text>
        </Card>

        <Card>
          <Text style={styles.label}>API Key</Text>
          <TextInput
            style={styles.input}
            value={apiKey}
            onChangeText={setApiKeyInput}
            autoCapitalize="none"
            autoCorrect={false}
            secureTextEntry
            placeholder="leave blank for local dev"
            placeholderTextColor={colors.textSecondary}
          />
          <Text style={styles.hint}>
            Only needed once the backend is deployed somewhere public with an API_KEY
            configured — local dev against a server with no API_KEY set ignores this
            entirely. Sent as the X-API-Key header on every request; never leave this
            blank once the backend is reachable from the internet.
          </Text>
        </Card>

        <Card>
          <View style={styles.switchRow}>
            <Text style={styles.label}>Deadline reminder</Text>
            <Switch value={reminderEnabled} onValueChange={setReminderEnabledInput} />
          </View>
          {reminderEnabled && (
            <>
              <Text style={[styles.label, styles.subLabel]}>Hours before deadline</Text>
              <TextInput
                style={styles.input}
                value={reminderHours}
                onChangeText={setReminderHoursInput}
                keyboardType="number-pad"
                placeholder={String(DEFAULT_REMINDER_HOURS_BEFORE)}
                placeholderTextColor={colors.textSecondary}
              />
            </>
          )}
          <Text style={styles.hint}>
            A local notification scheduled on this device — re-scheduled every time you save
            here or open the Squad tab, so it always tracks the CURRENT next deadline rather
            than a stale one from last week.
          </Text>
        </Card>

        <PrimaryButton label={savedFlash ? "Saved ✓" : "Save"} onPress={save} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: colors.background },
  container: { flex: 1 },
  content: { padding: spacing.md },
  label: { ...type.subtitle, marginBottom: spacing.xs },
  input: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.sm,
    padding: 12,
    fontSize: 16,
    backgroundColor: colors.background,
    color: colors.textPrimary,
  },
  hint: { ...type.caption, marginTop: spacing.sm, lineHeight: 16 },
  switchRow: {
    flexDirection: "row", justifyContent: "space-between",
    alignItems: "center", marginBottom: spacing.xs,
  },
  subLabel: { fontSize: 13, marginTop: spacing.sm, marginBottom: spacing.xs },
});
