#define MyAppName "BC Vision"
#define MyAppVersion "2.2.0-rc31"
#define MyAppPublisher "Gilas Abi Alborz"
#define MyAppExeName "BCVision.exe"
#define RequireExistingInstall 1

[Setup]
AppId={{12FC1F39-4F29-4D61-A81D-66BD900AA4E8}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
VersionInfoVersion=2.2.0.300
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\BC Vision
DisableDirPage=auto
DisableProgramGroupPage=yes
UsePreviousAppDir=yes
OutputDir=..\setup_output
OutputBaseFilename=BCVision_RC30_Update
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no
SetupLogging=yes
UninstallDisplayIcon={app}\{#MyAppExeName}

[Files]
Source: "..\dist\BCVision\*"; DestDir: "{app}"; Excludes: "runtime\current.txt,runtime\last-known-good.txt,runtime\previous.txt,runtime\pending.txt,runtime\failed.txt"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
Type: files; Name: "{app}\license_public_key.pem"
Type: files; Name: "{app}\app\license.py"
Type: files; Name: "{app}\app\license_format.py"
Type: files; Name: "{app}\app\offline_license_policy.py"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Run BC Vision"; Flags: nowait postinstall skipifsilent

[Code]
#include "Runtime_Pointer_Guard.iss"
