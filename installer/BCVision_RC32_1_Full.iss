#define MyAppVersion "2.2.0-rc32.1"

[Setup]
AppId={{12FC1F39-4F29-4D61-A81D-66BD900AA4E8}
AppName=BC Vision
AppVersion={#MyAppVersion}
VersionInfoVersion=2.2.0.321
AppPublisher=Gilas Abi Alborz
DefaultDirName={autopf}\BC Vision
UsePreviousAppDir=yes
DisableProgramGroupPage=yes
Uninstallable=no
CreateUninstallRegKey=no
OutputDir=..\setup_output
OutputBaseFilename=BCVision_RC32.1_Setup
Compression=none
SolidCompression=no
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no
SetupLogging=yes
ExtraDiskSpaceRequired=2147483648

[Files]
Source: "..\bundle_inputs\BCVision_RC32_Setup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall
Source: "..\bundle_inputs\BCVision_RC32.1_Update.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Run]
Filename: "{app}\BCVision.exe"; Description: "Run BC Vision"; Flags: nowait postinstall skipifsilent runasoriginaluser

[Code]
procedure RunChecked(FileName: String; Parameters: String; Description: String);
var
  ResultCode: Integer;
begin
  WizardForm.StatusLabel.Caption := Description;
  ResultCode := -1;
  if not Exec(FileName, Parameters, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    RaiseException(Description + ' could not start. See the setup log.');
  if ResultCode <> 0 then
    RaiseException(Description + ' failed with exit code ' + IntToStr(ResultCode) +
      '. The complete RC32.1 installation was not confirmed. See the setup log.');
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  BaseSetup: String;
  UpdateSetup: String;
  Parameters: String;
  VerifyData: String;
  StopCode: Integer;
begin
  if CurStep <> ssPostInstall then Exit;
  BaseSetup := ExpandConstant('{tmp}\BCVision_RC32_Setup.exe');
  UpdateSetup := ExpandConstant('{tmp}\BCVision_RC32.1_Update.exe');
  { These are the exact already-released and installed-tested binaries. }
  if CompareText(GetSHA256OfFile(BaseSetup),
    'd332a4e2bb3265314ea34d170f958dc82974ec743deb40c2f16d8a58d1bc4e72') <> 0 then
    RaiseException('The embedded full runtime failed SHA-256 verification.');
  if CompareText(GetSHA256OfFile(UpdateSetup),
    '542c07d4bdfac14050941cdda47277a219da74e97f9f2cfd1c5c19246c3a524f') <> 0 then
    RaiseException('The embedded RC32.1 updater failed SHA-256 verification.');
  Parameters := '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /DIR=' +
    AddQuotes(ExpandConstant('{app}'));
  { The original installers retain their stop, compatibility and rollback guards. }
  RunChecked(BaseSetup, Parameters + ' /LOG=' +
    AddQuotes(ExpandConstant('{%TEMP}\BCVision_RC32.1_base_setup.log')),
    'Installing the complete BC Vision runtime...');
  RunChecked(UpdateSetup, Parameters + ' /LOG=' +
    AddQuotes(ExpandConstant('{%TEMP}\BCVision_RC32.1_update_setup.log')),
    'Installing and verifying BC Vision RC32.1...');
  { Close the child updater's auto-launch; the outer setup owns desktop launch. }
  Exec(ExpandConstant('{cmd}'), '/C taskkill /IM BCVision.exe /T /F >nul 2>&1',
    '', SW_HIDE, ewWaitUntilTerminated, StopCode);
  VerifyData := ExpandConstant('{tmp}\bcvision-full-verification');
  ForceDirectories(VerifyData);
  RunChecked(ExpandConstant('{app}\BCVision.exe'),
    '--self-test --verify-anpr --runtime-candidate {#MyAppVersion}' +
    ' --self-test-data-dir ' + AddQuotes(VerifyData) +
    ' --self-test-output ' + AddQuotes(VerifyData + '\result.json'),
    'Verifying installed application and offline ANPR models...');
end;
