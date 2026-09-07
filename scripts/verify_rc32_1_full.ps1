$ErrorActionPreference = 'Stop'

function Invoke-Checked {
  param([string]$Path, [string[]]$Arguments, [int]$Timeout = 600)
  $process = Start-Process $Path -ArgumentList $Arguments -PassThru
  if (-not $process.WaitForExit($Timeout * 1000)) {
    & taskkill.exe /PID $process.Id /T /F
    throw "Process timed out: $Path"
  }
  $process.WaitForExit()
  if ($process.ExitCode -ne 0) { throw "$Path exited with $($process.ExitCode)" }
}

function Test-Installed {
  param([string]$InstallDir, [string]$DataDir, [string]$Label)
  $resultFile = Join-Path $env:RUNNER_TEMP "$Label.json"
  Invoke-Checked "$InstallDir\BCVision.exe" @(
    '--self-test', '--verify-anpr', '--self-test-data-dir', "`"$DataDir`"",
    '--self-test-output', "`"$resultFile`""
  ) 180
  $result = Get-Content $resultFile -Raw | ConvertFrom-Json
  if (-not $result.ok -or -not $result.anpr_ready -or
      $result.version -ne '2.2.0-rc32.1' -or
      $result.runtime_version -ne '2.2.0-rc32.1' -or
      $result.runtime_source -ne 'external') {
    throw "Installed full bundle failed application/model/version validation: $Label"
  }
  python scripts/verify_installed_runtime.py --install-dir $InstallDir `
    --version 2.2.0-rc32.1 --abi 2 --last-known-good 2.2.0-rc32.1
  if ($LASTEXITCODE -ne 0) { throw 'Installed runtime file validation failed' }
  Write-Host "PASS: $Label, RC32.1, offline ANPR and runtime files"
}

$bundle = (Resolve-Path 'setup_output/BCVision_RC32.1_Setup.exe').Path
$installDir = Join-Path $env:RUNNER_TEMP 'BC Vision Full Clean'
$dataDir = Join-Path $env:RUNNER_TEMP 'bcvision-full-clean-data'
Invoke-Checked $bundle @('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
  "/DIR=`"$installDir`"", "/LOG=`"$env:RUNNER_TEMP\full-clean.log`"")
Test-Installed $installDir $dataDir 'full-clean'
$uninstall = (Get-ChildItem $installDir -Filter 'unins*.exe' | Select-Object -First 1).FullName
if (-not $uninstall) { throw 'The complete application has no uninstaller' }
Invoke-Checked $uninstall @('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART')
if (-not (Test-Path "$dataDir/bcvision.db")) { throw 'Uninstall removed user data' }

# Upgrade a genuinely incompatible old full installation, with custom storage.
$installDir = Join-Path $env:RUNNER_TEMP 'BC Vision Full Upgrade'
$dataDir = Join-Path $env:RUNNER_TEMP 'bcvision-full-upgrade-data'
$bootstrapDir = Join-Path $env:RUNNER_TEMP 'bcvision-full-bootstrap'
$old = (Resolve-Path 'bundle_inputs/BCVision_RC28.1_Setup.exe').Path
Invoke-Checked $old @('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', "/DIR=`"$installDir`"")
New-Item -ItemType Directory $dataDir,$bootstrapDir -Force | Out-Null
$env:BCVISION_DATA_DIR = $dataDir
$first = Join-Path $env:RUNNER_TEMP 'full-old.json'
Invoke-Checked "$installDir\BCVision.exe" @('--self-test', '--self-test-output', "`"$first`"") 180
$before = Get-Content $first -Raw | ConvertFrom-Json
if (-not $before.ok -or $before.version -ne '2.2.0-rc28.1') { throw 'Old installation fixture failed' }
@{storage_root=$dataDir} | ConvertTo-Json | Set-Content "$bootstrapDir/storage_config.json" -Encoding utf8
$pointerHash = (Get-FileHash "$bootstrapDir/storage_config.json").Hash
$env:BCVISION_DATA_DIR = $bootstrapDir
$env:BCVISION_TEST_DB = Join-Path $dataDir 'bcvision.db'
python -c "import os,sqlite3; c=sqlite3.connect(os.environ['BCVISION_TEST_DB']); c.execute('INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)',('full_bundle_marker','preserved-rc32.1')); c.commit(); c.close()"
if ($LASTEXITCODE -ne 0) { throw 'Could not seed persistent database marker' }
$markers = @('videos/preserve.marker','snapshots/preserve.marker','plates/preserve.marker','models/plate/preserve.marker')
foreach ($relative in $markers) {
  $path = Join-Path $dataDir $relative
  New-Item -ItemType Directory (Split-Path $path) -Force | Out-Null
  Set-Content $path 'preserved-rc32.1'
}
Invoke-Checked $bundle @('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
  "/DIR=`"$installDir`"", "/LOG=`"$env:RUNNER_TEMP\full-upgrade.log`"")
Test-Installed $installDir $bootstrapDir 'full-upgrade'
python -c "import os,sqlite3; c=sqlite3.connect(os.environ['BCVISION_TEST_DB']); assert c.execute('SELECT value FROM settings WHERE key=?',('full_bundle_marker',)).fetchone()==('preserved-rc32.1',); c.close()"
if ($LASTEXITCODE -ne 0) { throw 'Full bundle changed database settings' }
if ((Get-FileHash "$bootstrapDir/storage_config.json").Hash -ne $pointerHash) { throw 'Custom storage pointer changed' }
foreach ($relative in $markers) {
  if ((Get-Content (Join-Path $dataDir $relative) -Raw).Trim() -ne 'preserved-rc32.1') { throw "Persistent media/model marker changed: $relative" }
}
Write-Host 'PASS: RC28.1 to RC32.1 upgrade preserved database, custom storage, media and model markers'
