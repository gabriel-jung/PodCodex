/** Platform abstraction interfaces — web vs Tauri implementations. */

export interface PlatformFS {
  /** Open a native folder picker dialog. Returns path or null if cancelled. */
  openFolderDialog(): Promise<string | null>;
  /** Open a native file picker dialog. Returns path or null if cancelled. */
  openFileDialog(extensions?: string[]): Promise<string | null>;
  /**
   * Open a native save-file dialog. Returns the chosen destination path or
   * null if the user cancelled.
   */
  saveFileDialog(opts?: {
    defaultPath?: string;
    extensions?: string[];
  }): Promise<string | null>;
}

export interface PlatformWindow {
  /** Set the window title bar text. */
  setTitle(title: string): void;
  /** Minimize the window. */
  minimize(): void;
  /** Whether this is a native (Tauri) window. */
  isNative(): boolean;
}

export interface PlatformLifecycle {
  /** Register a callback just before the app closes.
   *
   *  Best-effort and synchronous in spirit: neither implementation waits for
   *  async work (`beforeunload` cannot, and Tauri must not — see `tauri.ts`),
   *  so anything that must survive has to be written earlier. */
  onBeforeClose(cb: () => void): () => void;
}

export interface Platform {
  fs: PlatformFS;
  window: PlatformWindow;
  lifecycle: PlatformLifecycle;
  isTauri: boolean;
}
