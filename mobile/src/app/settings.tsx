import { useEffect, useState } from "react";
import { View, Text, TextInput, Pressable, StyleSheet, Alert } from "react-native";
import { router } from "expo-router";

import {
  DEFAULT_API_BASE_URL,
  getApiBaseUrl,
  setApiBaseUrl,
  getTeamId,
  setTeamId,
} from "@/lib/config";

export default function SettingsScreen() {
  const [apiBaseUrl, setApiBaseUrlInput] = useState(DEFAULT_API_BASE_URL);
  const [teamId, setTeamIdInput] = useState("");
  const [loaded, setLoaded] = useState(false);

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
    router.back();
  }

  if (!loaded) return null;

  return (
    <View style={styles.container}>
      <Text style={styles.label}>API base URL</Text>
      <TextInput
        style={styles.input}
        value={apiBaseUrl}
        onChangeText={setApiBaseUrlInput}
        autoCapitalize="none"
        autoCorrect={false}
        placeholder={DEFAULT_API_BASE_URL}
      />
      <Text style={styles.hint}>
        localhost only works from the iOS Simulator. On a physical phone, use your dev
        machine&apos;s LAN IP (e.g. http://192.168.1.42:8000) — the server must be started with
        `uvicorn api:app --host 0.0.0.0`. On the Android emulator, use http://10.0.2.2:8000.
      </Text>

      <Text style={styles.label}>FPL Team ID</Text>
      <TextInput
        style={styles.input}
        value={teamId}
        onChangeText={setTeamIdInput}
        keyboardType="number-pad"
        placeholder="e.g. 7362936"
      />
      <Text style={styles.hint}>The number in your FPL team&apos;s URL.</Text>

      <Pressable style={styles.button} onPress={save}>
        <Text style={styles.buttonText}>Save</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, gap: 4 },
  label: { fontWeight: "600", marginTop: 16 },
  input: {
    borderWidth: 1, borderColor: "#ccc", borderRadius: 8,
    padding: 10, marginTop: 4, fontSize: 16,
  },
  hint: { color: "#666", fontSize: 12, marginTop: 4 },
  button: {
    backgroundColor: "#37003c", borderRadius: 8, padding: 14,
    alignItems: "center", marginTop: 28,
  },
  buttonText: { color: "white", fontWeight: "600", fontSize: 16 },
});
