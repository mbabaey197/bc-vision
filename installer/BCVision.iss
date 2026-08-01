#define MyAppName "BC Vision"
#define MyAppVersion "2.2.0-rc21"
#define MyAppPublisher "Gilas Abi Alborz"
#define MyAppExeName "BCVision.exe"

[Setup]
AppId={{12FC1F39-4F29-4D61-A81D-66BD900AA4E8}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
VersionInfoVersion=2.2.0.21
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\BC Vision
DefaultGroupName=BC Vision
DisableProgramGroupPage=yes
UsePreviousAppDir=yes
OutputDir=..\setup_output
OutputBaseFilename=BCVision_Setup_v2.2.0-rc21
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=commandline
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no
SetupLogging=yes
UninstallDisplayIcon={app}\{#MyAppExeName}

[Files]
Source: "..\dist\BCVision\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\BC Vision"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\BC Vision"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Run BC Vision"; Flags: nowait postinstall skipifsilent
