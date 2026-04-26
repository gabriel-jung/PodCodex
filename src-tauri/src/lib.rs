use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use tauri::{Manager, RunEvent};

const API_PORT: u16 = 18811;
const HEALTH_TIMEOUT_SECS: u64 = 180;

/// Tauri externalBin sidecar basename. Project tree carries
/// ``src-tauri/binaries/podcodex-server-<triple>``; the bundler strips the
/// triple suffix when copying into the .app, leaving a bare ``podcodex-server``
/// alongside the host exe.
const SERVER_SIDECAR: &str = "podcodex-server";

struct BackendProcess(Mutex<Option<Child>>);

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
                    log::info!("Killing backend sidecar (pid={})", child.id());
                    if let Err(e) = child.kill() {
                        log::error!("Failed to kill backend sidecar: {e}");
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

    let server_exe = locate_sidecar(SERVER_SIDECAR).ok_or_else(|| {
        format!(
            "Cannot find {SERVER_SIDECAR} sidecar next to host binary. \
             Did `packaging/build_server.py` run before `cargo tauri build`?"
        )
    })?;

    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("cannot resolve app_data_dir: {e}"))?;
    std::fs::create_dir_all(&data_dir)?;

    let models_dir = data_dir.join("models");
    let hf_home = models_dir.join("huggingface");
    let torch_home = models_dir.join("torch");

    log::info!("Spawning backend: {:?}  (data_dir={:?})", server_exe, data_dir);

    let mut cmd = Command::new(&server_exe);
    cmd.env("PODCODEX_DATA_DIR", &data_dir)
        .env("PODCODEX_API_PORT", API_PORT.to_string())
        .env("HF_HOME", &hf_home)
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

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("backend spawn failed: {e}"))?;

    // Drain stdout/stderr on plain OS threads so we don't depend on a Tokio
    // reactor (Tauri's setup hook runs sync, before the runtime exists; using
    // tokio::process here panics with "no reactor running").
    if let Some(stdout) = child.stdout.take() {
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                log::info!("[backend] {}", line);
            }
        });
    }
    if let Some(stderr) = child.stderr.take() {
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
