import { useEffect, useState } from "react";
import { Text, TextInput, StyleSheet, Alert, ScrollView } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import {
  DEFAULT_API_BASE_URL,
  getApiBaseUrl,
  setApiBaseUrl,
  getTeamId,
  setTeamId,
} from "@/lib/config";
import { colors, spacing, radius, type } from "@/lib/theme";
import { Card, PrimaryButton } from "@/components/ui";

export default function SettingsScreen() {
  const [apiBaseUrl, setApiBaseUrlInput] = useState(DEFAULT_API_BASE_URL);
  const [teamId, setTeamIdInput] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [savedFlash, setSavedFlash] = useState(false);

  useEffect(() => {
    (async () => {
      setApiBaseUrlInput(await getApiBaseUrl());
      setTeamIdInput(await getTeamId());
      setLoaded(true);
    })();
  }, []);

  async function save() {
    if (teamId.trim() && Number.isNaN(Number(teamId.trim()))) {
      Alert.alert("Invalid Team ID", "Team ID must be a number — it's the number in your FPL team's URL.");
      return;
    }
    await setApiBaseUrl(apiBaseUrl);
    await setTeamId(teamId);
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
});
