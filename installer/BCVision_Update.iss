#define MyAppName "BC Vision"
#define MyAppVersion "2.2.0-rc5"
#define MyAppPublisher "Gilas Abi Alborz"
#define MyAppExeName "BCVision.exe"

[Setup]
AppId={{12FC1F39-4F29-4D61-A81D-66BD900AA4E8}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
VersionInfoVersion=2.2.0.5
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\BC Vision
DisableDirPage=auto
DisableProgramGroupPage=yes
UsePreviousAppDir=yes
OutputDir=..\setup_output
OutputBaseFilename=BCVision_Update_v2.2.0-rc5
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
Source: "..\dist\BCVision\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Run BC Vision"; Flags: nowait postinstall skipifsilent

[Code]
function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  if not FileExists(ExpandConstant('{app}\BCVision.exe')) then
    Result := 'BC Vision is not installed. Run the full installer first.'
  else
    Result := '';
end;
