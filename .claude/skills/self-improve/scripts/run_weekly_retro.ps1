# Weekly BiffBet self-improvement retro.
# Register once (from an elevated or normal PowerShell prompt):
#
#   schtasks /Create /TN "BiffBet weekly retro" /SC WEEKLY /D MON /ST 09:00 `
#     /TR "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\jim\mlb_value_bot\.claude\skills\self-improve\scripts\run_weekly_retro.ps1"
#
# Monday 9am ET: Sunday's slate has settled and graded overnight, so the
# week's data is complete when this fires.

$repo = "C:\Users\jim\mlb_value_bot"
Set-Location $repo

# 1. Deterministic data collection (read-only; caches pass grades).
& "$repo\.venv\Scripts\python.exe" "$repo\.claude\skills\self-improve\scripts\retro_analysis.py" `
    *> "$repo\storage\retro\last_run.log"

# 2. Interpretation + ledger update via Claude Code (headless).
#    Comment this block out if you prefer to run /self-improve manually
#    after reviewing storage\retro\retro_report_<date>.md yourself.
$claude = Get-Command claude -ErrorAction SilentlyContinue
if ($claude) {
    & claude -p "/self-improve — the retro script already ran; start from step 2 (reconcile the newest storage/retro report against docs/abilities/ledger.json). Remember: propose only, never implement." `
        --permission-mode acceptEdits `
        *>> "$repo\storage\retro\last_run.log"
} else {
    Add-Content "$repo\storage\retro\last_run.log" "claude CLI not found - run /self-improve manually to interpret the report."
}

# 3. Email alert (jim@financewithjim.com) when a trend cleared the bar or a
#    proposal is awaiting approval. Needs RETRO_SMTP_* in .env; skips politely
#    (and logs) when they are absent. Runs after step 2 so freshly written
#    proposals are included.
& "$repo\.venv\Scripts\python.exe" "$repo\.claude\skills\self-improve\scripts\notify_trends.py" `
    *>> "$repo\storage\retro\last_run.log"
