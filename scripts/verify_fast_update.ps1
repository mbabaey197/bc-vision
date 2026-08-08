[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$UpdatePath,
    [Parameter(Mandatory = $true)]
    [string]$DistPath,
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [Parameter(Mandatory = $true)]
    [string]$WorkRoot
)

$ErrorActionPreference = "Stop"
$installDir = Join-Path $WorkRoot "bcvision-fast-install"
$dataDir = Join-Path $WorkRoot "bcvision-fast-data"
$firstResult = Join-Path $WorkRoot "fast-self-test-before.json"
$updatedResult = Join-Path $WorkRoot "fast-self-test-after.json"
$updateLog = Join-Path $WorkRoot "bcvision-fast-update.log"

foreach ($path in @($installDir, $dataDir)) {
    if (Test-Path $path) {
        Remove-Item $path -Recurse -Force
    }
}
New-Item -ItemType Directory -Path $installDir -Force | Out-Null
Copy-Item (Join-Path $DistPath "*") $installDir -Recurse -Force

$executable = Join-Path $installDir "BCVision.exe"
if (-not (Test-Path $executable)) {
    throw "BCVision.exe was not copied into the fast-update sandbox"
}

$env:BCVISION_DATA_DIR = $dataDir
$env:BCVISION_SKIP_MODEL_PREP = "1"
$first = Start-Process -FilePath $executable -ArgumentList @(
    "--self-test",
    "--verify-anpr",
    "--verify-no-license",
    "--self-test-output",
    $firstResult
) -Wait -PassThru
if ($first.ExitCode -ne 0) {
    throw "Pre-update executable self-test exited with $($first.ExitCode)"
}
$firstJson = Get-Content $firstResult -Raw | ConvertFrom-Json
if (
    -not $firstJson.ok -or
    -not $firstJson.anpr_ready -or
    -not $firstJson.no_license_ready -or
    $firstJson.version -ne $Version
) {
    throw "Pre-update executable self-test failed"
}

$marker = "preserve-$([guid]::NewGuid())"
$env:BCVISION_TEST_MARKER = $marker
python -c "import os,sqlite3; p=os.path.join(os.environ['BCVISION_DATA_DIR'],'bcvision.db'); c=sqlite3.connect(p); c.execute('INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)',('release_test_marker',os.environ['BCVISION_TEST_MARKER'])); c.commit(); c.close()"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to create the database preservation marker"
}
$modelDir = Join-Path $dataDir "models\plate"
New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
$modelMarker = Join-Path $modelDir "preserve.marker"
Set-Content -Path $modelMarker -Value $marker

$update = Start-Process -FilePath (Resolve-Path $UpdatePath) -ArgumentList @(
    "/VERYSILENT",
    "/SUPPRESSMSGBOXES",
    "/NORESTART",
    "/CURRENTUSER",
    "/LOG=$updateLog",
    "/DIR=$installDir"
) -Wait -PassThru
if ($update.ExitCode -ne 0) {
    if (Test-Path $updateLog) {
        Get-Content $updateLog
    }
    throw "Updater exited with $($update.ExitCode)"
}

$updated = Start-Process -FilePath $executable -ArgumentList @(
    "--self-test",
    "--verify-anpr",
    "--verify-no-license",
    "--self-test-output",
    $updatedResult
) -Wait -PassThru
if ($updated.ExitCode -ne 0) {
    throw "Updated executable self-test exited with $($updated.ExitCode)"
}
$updatedJson = Get-Content $updatedResult -Raw | ConvertFrom-Json
if (
    -not $updatedJson.ok -or
    -not $updatedJson.anpr_ready -or
    -not $updatedJson.no_license_ready -or
    $updatedJson.version -ne $Version
) {
    throw "Updated executable self-test failed"
}

python -c "import os,sqlite3,sys; p=os.path.join(os.environ['BCVISION_DATA_DIR'],'bcvision.db'); c=sqlite3.connect(p); r=c.execute('SELECT value FROM settings WHERE key=?',('release_test_marker',)).fetchone(); c.close(); sys.exit(0 if r and r[0]==os.environ['BCVISION_TEST_MARKER'] else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Database marker was not preserved by the updater"
}
if ((Get-Content $modelMarker -Raw).Trim() -ne $marker) {
    throw "AI model marker was not preserved by the updater"
}

Write-Output "Fast updater smoke test passed for $Version"
