/**
 * Deadline reminder — a LOCAL scheduled notification (not push/remote),
 * which is exactly why this needs no backend involvement beyond reading
 * /deadline: expo-notifications' local notifications work fully inside
 * Expo Go on both iOS and Android (only remote push dropped Expo Go
 * support on Android from SDK 53+, and this app doesn't use that).
 *
 * This app only ever needs ONE active reminder (the next deadline), so
 * every (re)schedule cancels whatever was previously scheduled first —
 * no per-notification id bookkeeping needed.
 */
import * as Notifications from "expo-notifications";
import { Platform } from "react-native";

import { getApiBaseUrl, getReminderHoursBefore } from "@/lib/config";
import { getDeadline } from "@/lib/api";

Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowBanner: true,
    shouldShowList: true,
    shouldPlaySound: false,
    shouldSetBadge: false,
  }),
});

export async function requestNotificationPermission(): Promise<boolean> {
  const current = await Notifications.getPermissionsAsync();
  if (current.granted) return true;
  const requested = await Notifications.requestPermissionsAsync();
  return requested.granted;
}

/**
 * Fetches the current deadline and schedules a reminder `hoursBefore`
 * hours ahead of it. Returns the Date it scheduled for, or null if the
 * computed reminder time has already passed (e.g. hoursBefore is bigger
 * than the time actually left) — callers should treat null as "nothing
 * to remind about right now," not an error.
 */
export async function scheduleDeadlineReminder(): Promise<Date | null> {
  const baseUrl = await getApiBaseUrl();
  const hoursBefore = await getReminderHoursBefore();
  const { gw, deadline_time } = await getDeadline(baseUrl);

  await Notifications.cancelAllScheduledNotificationsAsync();

  if (!deadline_time) return null;
  const deadline = new Date(deadline_time);
  const reminderAt = new Date(deadline.getTime() - hoursBefore * 60 * 60 * 1000);
  if (reminderAt.getTime() <= Date.now()) return null;

  await Notifications.scheduleNotificationAsync({
    content: {
      title: `GW${gw} deadline in ${hoursBefore}h`,
      body: "Check your squad before the deadline locks it in.",
    },
    trigger: {
      type: Notifications.SchedulableTriggerInputTypes.DATE,
      date: reminderAt,
      channelId: Platform.OS === "android" ? "deadline-reminders" : undefined,
    },
  });
  return reminderAt;
}

export async function cancelDeadlineReminder(): Promise<void> {
  await Notifications.cancelAllScheduledNotificationsAsync();
}

/** Android requires an explicit notification channel — a no-op on iOS. */
export async function ensureAndroidChannel(): Promise<void> {
  if (Platform.OS !== "android") return;
  await Notifications.setNotificationChannelAsync("deadline-reminders", {
    name: "Deadline reminders",
    importance: Notifications.AndroidImportance.DEFAULT,
  });
}
