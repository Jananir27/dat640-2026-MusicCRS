// This random ID identifies one browser profile without an account.
export function getDeviceId(): string {
  const key = "musiccrs.deviceId";
  let id = window.localStorage.getItem(key);
  if (!id || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id)) {
    id = window.crypto.randomUUID();
    window.localStorage.setItem(key, id);
  }
  return id;
}
