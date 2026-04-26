#!/usr/bin/env bash
# Sign + notarize the bundled .app for macOS distribution.
#
# Tauri's `cargo tauri build` already codesigns when APPLE_SIGNING_IDENTITY
# is set; this wrapper centralises the env it expects so the user only has
# to fill in the keychain profile / cert names once.
#
# Prereqs:
#   1. Apple Developer ID Application certificate installed in login keychain
#   2. App-specific password for notarytool stored in keychain via:
#        xcrun notarytool store-credentials podcodex-notary \
#          --apple-id you@example.com --team-id ABCDEF1234 --password app-specific-pwd
#   3. Set APPLE_SIGNING_IDENTITY to the cert's common name, e.g.
#        export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (ABCDEF1234)"
#
# Usage:
#   scripts/sign_and_notarize.sh           # build + sign + notarize + staple
#   scripts/sign_and_notarize.sh --no-notary  # sign only (faster iteration)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NOTARY_KEYCHAIN_PROFILE="${PODCODEX_NOTARY_PROFILE:-podcodex-notary}"
APP_NAME="PodCodex"
BUNDLE_DIR="$REPO_ROOT/src-tauri/target/release/bundle/macos"
APP_PATH="$BUNDLE_DIR/$APP_NAME.app"

run_notary=true
if [[ "${1:-}" == "--no-notary" ]]; then
    run_notary=false
fi

if [[ -z "${APPLE_SIGNING_IDENTITY:-}" ]]; then
    echo "ERROR: APPLE_SIGNING_IDENTITY not set." >&2
    echo "  export APPLE_SIGNING_IDENTITY='Developer ID Application: Your Name (TEAMID)'" >&2
    exit 1
fi

echo "==> Building (signs during build via cargo tauri build)"
cd "$REPO_ROOT"
make bundle

if [[ ! -d "$APP_PATH" ]]; then
    echo "ERROR: bundle not produced at $APP_PATH" >&2
    exit 1
fi

echo "==> Verifying signature"
codesign --verify --deep --strict --verbose=2 "$APP_PATH"

# Each PyInstaller-bundled .so / .dylib needs its own signature under hardened
# runtime. cargo tauri build signs the outer .app, but nested PyInstaller
# binaries shipped via externalBin sometimes get missed. Re-sign defensively.
echo "==> Re-signing nested binaries"
find "$APP_PATH/Contents" -type f \( -name "*.so" -o -name "*.dylib" \) -print0 | \
    xargs -0 -I {} codesign --force --sign "$APPLE_SIGNING_IDENTITY" \
        --options runtime --timestamp \
        --entitlements "$REPO_ROOT/src-tauri/Entitlements.plist" "{}"

# Re-sign the outer .app to update the bundle seal after nested re-signs.
codesign --force --deep --sign "$APPLE_SIGNING_IDENTITY" \
    --options runtime --timestamp \
    --entitlements "$REPO_ROOT/src-tauri/Entitlements.plist" "$APP_PATH"

if ! $run_notary; then
    echo "==> Skipping notarization (--no-notary)"
    exit 0
fi

ZIP_PATH="$BUNDLE_DIR/$APP_NAME.zip"
echo "==> Zipping for notary submission"
ditto -c -k --keepParent "$APP_PATH" "$ZIP_PATH"

echo "==> Submitting to notary (this can take 5-15 min)"
xcrun notarytool submit "$ZIP_PATH" \
    --keychain-profile "$NOTARY_KEYCHAIN_PROFILE" \
    --wait

echo "==> Stapling notarization ticket"
xcrun stapler staple "$APP_PATH"

echo "==> Verifying staple"
xcrun stapler validate "$APP_PATH"
spctl -a -t exec -vv "$APP_PATH"

rm -f "$ZIP_PATH"
echo
echo "Done. Signed + notarized bundle at: $APP_PATH"
