use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Mutex;

use command_group::{CommandGroup, GroupChild};
use tauri::{Manager, RunEvent};

const API_PORT: u16 = 18811;
const HEALTH_TIMEOUT_SECS: u64 = 180;

/// Tauri externalBin sidecar basename. Project tree carries
/// ``src-tauri/binaries/podcodex-server-<triple>``; the bundler strips the
/// triple suffix when copying into the .app, leaving a bare ``podcodex-server``
/// alongside the host exe.
const SERVER_SIDECAR: &str = "podcodex-server";

/// Optional GPU sidecar — installed at runtime into <app_data>/backends/gpu/
/// when the user opts in via Settings. See ``packaging/package_gpu.py`` for
/// the archive layout and ``src/podcodex/api/gpu_backend.py`` for the
/// download/extract logic.
const GPU_BACKEND_SUBDIR: &str = "backends/gpu";
const GPU_SIDECAR_NAME: &str = "podcodex-server-gpu";
const GPU_ACTIVATED_MARKER: &str = "activated";
const GPU_MANIFEST_FILE: &str = "cuda-libs.json";

struct BackendProcess(Mutex<Option<GroupChild>>);

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(BackendProcess(Mutex::new(None)))
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .setup(|app| {
            // Init log plugin in both debug and release. Backend draining
            // threads use log::info! / log::warn! and we want those visible
            // in Console.app for shipped builds too.
            app.handle().plugin(
                tauri_plugin_log::Builder::default()
                    .level(log::LevelFilter::Info)
                    .targets([
                        tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::Stdout),
                        tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::LogDir { file_name: None }),
                    ])
                    .build(),
            )?;

            spawn_backend_if_needed(app.handle())?;
            schedule_window_show(app.handle().clone());

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let RunEvent::Exit = event {
                let state = app_handle.state::<BackendProcess>();
                let child = state.0.lock().unwrap().take();
                if let Some(mut child) = child {
                    // GroupChild::kill terminates the entire job/process group,
                    // so multiprocessing workers die with the parent.
                    log::info!("Killing backend process group (pid={})", child.id());
                    if let Err(e) = child.kill() {
                        log::error!("Failed to kill backend process group: {e}");
                    }
                }
            }
        });
}

/// Spawn the bundled podcodex-server unless ``PODCODEX_SKIP_BACKEND_SPAWN``
/// is set (dev mode where ``make dev-api`` runs uvicorn from .venv).
fn spawn_backend_if_needed(app: &tauri::AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("PODCODEX_SKIP_BACKEND_SPAWN").is_ok() {
        log::info!(
            "PODCODEX_SKIP_BACKEND_SPAWN set, expecting external backend on :{}",
            API_PORT
        );
        return Ok(());
    }

    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("cannot resolve app_data_dir: {e}"))?;
    std::fs::create_dir_all(&data_dir)?;

    // Prefer the user-installed GPU sidecar if present, activated, and
    // version-matched; fall back to the bundled CPU sidecar otherwise. The
    // two paths are interchangeable from FastAPI's perspective — both expose
    // the same /api/* surface — but the GPU build has CUDA torch + cuDNN
    // bundled for hardware-accelerated transcription.
    //
    // Read from package_info (Cargo.toml-baked) rather than config (tauri.conf.json),
    // because we deliberately omit `version` from tauri.conf.json so it derives from
    // Cargo.toml — config().version is None at runtime in that setup.
    let app_version = app.package_info().version.to_string();
    let (server_exe, gpu_install_dir, backend_label) =
        match locate_gpu_sidecar(&data_dir, &app_version) {
            Some((binary, install_dir)) => (binary, Some(install_dir), "GPU"),
            None => {
                let cpu = locate_sidecar(SERVER_SIDECAR).ok_or_else(|| {
                    format!(
                        "Cannot find {SERVER_SIDECAR} sidecar next to host binary. \
                         Did `packaging/build_server.py` run before `cargo tauri build`?"
                    )
                })?;
                (cpu, None, "CPU")
            }
        };

    let models_dir = data_dir.join("models");
    let hf_home = models_dir.join("huggingface");
    let torch_home = models_dir.join("torch");

    log::info!(
        "Spawning {} backend: {:?}  (data_dir={:?})",
        backend_label,
        server_exe,
        data_dir
    );

    let mut cmd = Command::new(&server_exe);
    // PyInstaller --onedir (used by the GPU build) resolves _internal/ and
    // the bundled NVIDIA libs relative to cwd. Without setting current_dir,
    // the binary may fail to find its support files. CPU --onefile uses
    // _MEIPASS for all resolution and doesn't care about cwd, so we leave
    // it inherited from the Tauri parent in that case.
    if let Some(ref dir) = gpu_install_dir {
        cmd.current_dir(dir);
    }
    cmd.env("PODCODEX_DATA_DIR", &data_dir)
        .env("PODCODEX_API_PORT", API_PORT.to_string())
        // HF_HUB_CACHE is the canonical env var huggingface_hub respects for
        // its model cache. We don't set HF_HOME — that would tell HF Hub to
        // also look in <HF_HOME>/hub/, splitting the cache across two layouts
        // depending on which library entry-point downloaded.
        .env("HF_HUB_CACHE", hf_home.join("hub"))
        .env("TORCH_HOME", &torch_home)
        .env("TRANSFORMERS_CACHE", hf_home.join("transformers"))
        .env(
            "SENTENCE_TRANSFORMERS_HOME",
            models_dir.join("sentence-transformers"),
        )
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null());

    if let Some(ffmpeg) = locate_sidecar("ffmpeg") {
        log::info!("Bundled ffmpeg: {:?}", ffmpeg);
        cmd.env("FFMPEG_BINARY", ffmpeg);
    }
    if let Some(ytdlp) = locate_sidecar("yt-dlp") {
        log::info!("Bundled yt-dlp: {:?}", ytdlp);
        cmd.env("YT_DLP_BINARY", ytdlp);
    }

    // group_spawn wraps the child in a Windows Job Object (with
    // KILL_ON_JOB_CLOSE) or a Unix process group. Without this, multiprocessing
    // workers (torch DataLoader, whisperx, pyannote) survive parent kill on
    // app exit and accumulate as orphaned podcodex-server.exe processes.
    let mut child = cmd
        .group_spawn()
        .map_err(|e| format!("backend spawn failed: {e}"))?;

    // Drain stdout/stderr on plain OS threads so we don't depend on a Tokio
    // reactor (Tauri's setup hook runs sync, before the runtime exists; using
    // tokio::process here panics with "no reactor running").
    let inner = child.inner();
    if let Some(stdout) = inner.stdout.take() {
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                log::info!("[backend] {}", line);
            }
        });
    }
    if let Some(stderr) = inner.stderr.take() {
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                log::warn!("[backend] {}", line);
            }
        });
    }

    app.state::<BackendProcess>()
        .0
        .lock()
        .unwrap()
        .replace(child);

    Ok(())
}

/// Look for an installed + activated GPU sidecar at
/// ``<data_dir>/backends/gpu/podcodex-server-gpu``. Returns ``None`` (fall
/// back to CPU) on any of:
///   - activation marker missing
///   - manifest missing (broken/interrupted install)
///   - binary not present
///   - binary's reported version != ``app_version`` (ABI/contract drift
///     after an app update without a matching GPU re-download)
///
/// On success returns ``(binary_path, install_dir)`` so the caller can set
/// cwd correctly — PyInstaller --onedir needs cwd at the install root.
fn locate_gpu_sidecar(
    data_dir: &std::path::Path,
    app_version: &str,
) -> Option<(PathBuf, PathBuf)> {
    let install_dir = data_dir.join(GPU_BACKEND_SUBDIR);
    if !install_dir.join(GPU_ACTIVATED_MARKER).is_file() {
        return None;
    }
    if !install_dir.join(GPU_MANIFEST_FILE).is_file() {
        log::warn!(
            "GPU activated marker present but manifest missing — \
             ignoring activation, falling back to CPU"
        );
        return None;
    }
    let binary = {
        let bare = install_dir.join(GPU_SIDECAR_NAME);
        let exe = install_dir.join(format!("{GPU_SIDECAR_NAME}.exe"));
        if bare.is_file() {
            bare
        } else if exe.is_file() {
            exe
        } else {
            log::warn!(
                "GPU activated but sidecar binary not found at {:?}",
                install_dir
            );
            return None;
        }
    };

    match probe_sidecar_version(&binary, &install_dir) {
        Some(v) if v == app_version => Some((binary, install_dir)),
        Some(v) => {
            log::warn!(
                "GPU sidecar version {v} != app version {app_version}, \
                 falling back to CPU until user re-downloads"
            );
            None
        }
        None => {
            log::warn!(
                "Could not read GPU sidecar version, falling back to CPU"
            );
            None
        }
    }
}

/// Run ``<binary> --version`` (with ``cwd`` set to the install dir so
/// PyInstaller --onedir can find its support files) and return the trailing
/// version token from the output, e.g. ``"0.1.0"``.
fn probe_sidecar_version(
    binary: &std::path::Path,
    cwd: &std::path::Path,
) -> Option<String> {
    let output = Command::new(binary)
        .arg("--version")
        .current_dir(cwd)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    // Output format: "podcodex-server X.Y.Z\n"
    stdout
        .trim()
        .split_whitespace()
        .last()
        .map(|w| w.to_string())
}

/// Find a Tauri externalBin sidecar by short name. Bundled .app: bare
/// basename. ``cargo tauri dev``: ``<name>-<triple>``.
fn locate_sidecar(short_name: &str) -> Option<PathBuf> {
    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();

    let exact = exe_dir.join(short_name);
    if exact.is_file() {
        return Some(exact);
    }
    let exact_exe = exe_dir.join(format!("{short_name}.exe"));
    if exact_exe.is_file() {
        return Some(exact_exe);
    }

    let prefix = format!("{short_name}-");
    for entry in std::fs::read_dir(&exe_dir).ok()?.flatten() {
        let fname = entry.file_name();
        let fname = fname.to_string_lossy();
        if fname.starts_with(&prefix) {
            let path = entry.path();
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

/// Poll the backend's /api/health endpoint, then reveal the main window.
fn schedule_window_show(app: tauri::AppHandle) {
    tauri::async_runtime::spawn(async move {
        let window = match app.get_webview_window("main") {
            Some(w) => w,
            None => {
                log::error!("main window not found at startup");
                return;
            }
        };

        let client = reqwest::Client::new();
        let url = format!("http://127.0.0.1:{}/api/health", API_PORT);
        let deadline =
            tokio::time::Instant::now() + tokio::time::Duration::from_secs(HEALTH_TIMEOUT_SECS);

        loop {
            if tokio::time::Instant::now() >= deadline {
                log::warn!(
                    "Backend health check timed out after {}s — showing window anyway",
                    HEALTH_TIMEOUT_SECS
                );
                break;
            }
            if let Ok(resp) = client.get(&url).send().await {
                if resp.status().is_success() {
                    log::info!("Backend healthy");
                    break;
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;
        }

        let _ = window.show();
    });
}
