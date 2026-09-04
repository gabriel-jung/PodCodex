/** Test globals for the node environment.
 *
 *  The unit tests here cover pure logic and one localStorage migration, so a
 *  full DOM environment would be a heavy dependency for two globals. (It is
 *  also not a working one: vitest's jsdom environment exposes no
 *  `localStorage`, which is the single global the migration needs.)
 *
 *  `window` is here because importing almost anything under `src/api/`
 *  reaches `platform/isTauri.ts`, which reads `window` at module scope.
 */

class MemoryStorage implements Storage {
  private store = new Map<string, string>();

  get length(): number {
    return this.store.size;
  }
  clear(): void {
    this.store.clear();
  }
  getItem(key: string): string | null {
    return this.store.get(key) ?? null;
  }
  key(index: number): string | null {
    return [...this.store.keys()][index] ?? null;
  }
  removeItem(key: string): void {
    this.store.delete(key);
  }
  setItem(key: string, value: string): void {
    this.store.set(key, String(value));
  }
}

const storage = new MemoryStorage();

Object.assign(globalThis, {
  localStorage: storage,
  sessionStorage: new MemoryStorage(),
  window: {
    localStorage: storage,
    location: { origin: "http://localhost", href: "http://localhost/" },
    addEventListener: () => {},
    removeEventListener: () => {},
  },
});
