use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::{Manager, RunEvent};

struct BackendProcess(Mutex<Option<Child>>);

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let backend = BackendProcess(Mutex::new(None));

    // Spawn backend unless PODCODEX_SKIP_BACKEND_SPAWN is set
    let skip_spawn = std::env::var("PODCODEX_SKIP_BACKEND_SPAWN").is_ok();
    if !skip_spawn {
        // Resolve project root: parent of CARGO_MANIFEST_DIR in dev, or app dir in prod
        let project_root = if cfg!(debug_assertions) {
            let manifest_dir = env!("CARGO_MANIFEST_DIR");
            std::path::PathBuf::from(manifest_dir)
                .parent()
                .expect("CARGO_MANIFEST_DIR has no parent")
                .to_path_buf()
        } else {
            // In production, assume the binary is inside the project
            std::env::current_exe()
                .expect("cannot resolve exe path")
                .parent()
                .expect("exe has no parent dir")
                .to_path_buf()
        };

        let venv_uvicorn = project_root.join(".venv/bin/uvicorn");
        log::info!("Spawning backend: {:?}", venv_uvicorn);

        match Command::new(&venv_uvicorn)
            .args([
                "podcodex.api.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                "18811",
            ])
            .current_dir(&project_root)
            .spawn()
        {
            Ok(child) => {
                *backend.0.lock().unwrap() = Some(child);
            }
            Err(e) => {
                log::error!("Failed to spawn backend: {}", e);
            }
        }
    }

    tauri::Builder::default()
        .manage(backend)
        .setup(|app| {
            // Log plugin (debug only)
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            // Plugins
            app.handle().plugin(tauri_plugin_shell::init())?;
            app.handle().plugin(tauri_plugin_dialog::init())?;
            app.handle().plugin(tauri_plugin_window_state::Builder::default().build())?;

            // Health-check polling — show window once backend is ready
            let window = app
                .get_webview_window("main")
                .expect("main window not found");

            tauri::async_runtime::spawn(async move {
                let client = reqwest::Client::new();
                let url = "http://127.0.0.1:18811/api/health";
                let deadline =
                    tokio::time::Instant::now() + tokio::time::Duration::from_secs(15);

                loop {
                    if tokio::time::Instant::now() >= deadline {
                        log::warn!("Backend health check timed out after 15s — showing window anyway");
                        break;
                    }
                    match client.get(url).send().await {
                        Ok(resp) if resp.status().is_success() => {
                            log::info!("Backend is healthy");
                            break;
                        }
                        _ => {}
                    }
                    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
                }

                let _ = window.show();
            });

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let RunEvent::Exit = event {
                let state = app_handle.state::<BackendProcess>();
                let mut guard = state.0.lock().unwrap();
                if let Some(ref mut child) = *guard {
                    log::info!("Killing backend process");
                    let _ = child.kill();
                }
            }
        });
}
