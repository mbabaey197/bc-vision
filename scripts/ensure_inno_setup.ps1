[CmdletBinding()]
param(
    [string]$Version = "6.7.3"
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$knownPaths = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles}\Inno Setup 6\ISCC.exe"
) | Where-Object { $_ -and (Test-Path $_) }

if ($knownPaths.Count -gt 0) {
    Write-Output $knownPaths[0]
    exit 0
}

$cacheRoot = $env:RUNNER_TOOL_CACHE
if (-not $cacheRoot) {
    $cacheRoot = Join-Path $env:LOCALAPPDATA "BCVisionBuildTools"
}
$installDir = Join-Path $cacheRoot "bcvision\inno-setup-$Version"
$iscc = Join-Path $installDir "ISCC.exe"
if (Test-Path $iscc) {
    Write-Output $iscc
    exit 0
}

$downloadDir = Join-Path $cacheRoot "bcvision\downloads"
New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null
$installer = Join-Path $downloadDir "innosetup-$Version.exe"

if (-not (Test-Path $installer)) {
    $releaseVersion = $Version.Replace(".", "_")
    Invoke-WebRequest -UseBasicParsing `
        -Uri "https://github.com/jrsoftware/issrc/releases/download/is-$releaseVersion/innosetup-$Version.exe" `
        -OutFile $installer
}

$signature = Get-AuthenticodeSignature $installer
if (
    $signature.Status -ne "Valid" -or
    $signature.SignerCertificate.Subject -notmatch "Pyrsys B\.V\."
) {
    Remove-Item $installer -Force -ErrorAction SilentlyContinue
    throw "Inno Setup Authenticode verification failed"
}

$install = Start-Process -FilePath $installer -ArgumentList @(
    "/VERYSILENT",
    "/SUPPRESSMSGBOXES",
    "/NORESTART",
    "/CURRENTUSER",
    "/DIR=$installDir"
) -Wait -PassThru
if ($install.ExitCode -ne 0) {
    throw "Inno Setup installer exited with $($install.ExitCode)"
}
if (-not (Test-Path $iscc)) {
    throw "ISCC.exe was not installed"
}

Write-Output $iscc
