import { getDeviceId } from "../deviceIdentity";

test("reuses one anonymous ID until browser storage is cleared", () => {
  localStorage.clear();
  const randomUUID = jest.fn()
    .mockReturnValueOnce("1872e655-ddaa-4f97-913c-689286db4622")
    .mockReturnValueOnce("b7a26e0c-1904-4c6b-a4a0-9100206c810e");
  Object.defineProperty(window, "crypto", { value: { randomUUID }, configurable: true });
  const first = getDeviceId();
  expect(getDeviceId()).toBe(first);
  expect(randomUUID).toHaveBeenCalledTimes(1);
  localStorage.clear();
  expect(getDeviceId()).not.toBe(first);
});
