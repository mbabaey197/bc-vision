#ifndef MyAppVersion
  #error MyAppVersion must be supplied by the fast-update workflow
#endif
#ifndef MyVersionInfo
  #error MyVersionInfo must be supplied by the fast-update workflow
#endif
#ifndef MyReleaseLabel
  #error MyReleaseLabel must be supplied by the fast-update workflow
#endif
#ifndef MyRuntimeAbi
  #error MyRuntimeAbi must be supplied by the fast-update workflow
#endif
#ifndef MyRuntimeContract
  #error MyRuntimeContract must be supplied by the fast-update workflow
#endif
#ifndef MyBaseVersion
  #error MyBaseVersion must be supplied by the fast-update workflow
#endif

#define MyAppName "BC Vision"
#define MyAppPublisher "Gilas Abi Alborz"
#define MyAppExeName "BCVision.exe"

[Setup]
AppId={{12FC1F39-4F29-4D61-A81D-66BD900AA4E8}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
VersionInfoVersion={#MyVersionInfo}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\BC Vision
DisableDirPage=auto
DisableProgramGroupPage=yes
UsePreviousAppDir=yes
OutputDir=..\setup_output
OutputBaseFilename=BCVision_{#MyReleaseLabel}_Update
Compression=lzma2/fast
SolidCompression=no
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no
SetupLogging=yes
CreateUninstallRegKey=no
UninstallLogMode=append
UpdateUninstallLogAppName=no

[Files]
Source: "..\fast_update_payload\{#MyAppVersion}\app\*"; DestDir: "{app}\runtime\{#MyAppVersion}\app"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\fast_update_payload\{#MyAppVersion}\runtime-manifest.json"; DestDir: "{app}\runtime\{#MyAppVersion}"; Flags: ignoreversion

[InstallDelete]
Type: filesandordirs; Name: "{app}\runtime\{#MyAppVersion}"

[Run]
; The in-app updater is deliberately launched with /SILENT.  This entry must
; therefore run unconditionally after the candidate self-test commits, and it
; must return to the desktop user's token instead of keeping setup elevation.
Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Description: "Run BC Vision"; Flags: nowait runasoriginaluser

[Code]
const
  MoveFileReplaceExisting = 1;
  MoveFileWriteThrough = 8;

var
  RollbackVersion: String;

function MoveFileEx(
  ExistingFileName: String;
  NewFileName: String;
  Flags: Cardinal
): Boolean;
  external 'MoveFileExW@kernel32.dll stdcall setuponly';

function RuntimeRoot(): String;
begin
  Result := ExpandConstant('{app}\runtime');
end;

function MarkerPath(Name: String): String;
begin
  Result := RuntimeRoot() + '\' + Name;
end;

function AtomicWriteMarker(Name: String; Value: String): Boolean;
var
  Destination: String;
  Temporary: String;
begin
  Destination := MarkerPath(Name);
  Temporary := MarkerPath('.' + Name + '.update.tmp');
  DeleteFile(Temporary);
  Result := SaveStringToFile(
    Temporary,
    Value + #13#10,
    False
  );
  if Result then
    Result := MoveFileEx(
      Temporary,
      Destination,
      MoveFileReplaceExisting or MoveFileWriteThrough
    );
  if not Result then
    DeleteFile(Temporary);
end;

procedure DeleteMarker(Name: String);
begin
  DeleteFile(MarkerPath(Name));
  DeleteFile(MarkerPath('.' + Name + '.update.tmp'));
end;

function LoadMarker(Name: String): String;
var
  Value: AnsiString;
begin
  Result := '';
  if LoadStringFromFile(MarkerPath(Name), Value) then
    Result := Trim(Value);
end;

function NormalizeRuntimeVersion(
  Value: String;
  var Valid: Boolean
): String;
var
  Index: Integer;
begin
  Result := LowerCase(Trim(Value));
  Valid := StringChangeEx(Result, '-rc', '.', True) = 1;
  if not Valid or (Result = '') then
    Exit;
  if (Result[1] = '.') or (Result[Length(Result)] = '.') then
  begin
    Valid := False;
    Exit;
  end;
  for Index := 1 to Length(Result) do
  begin
    if not (
      ((Result[Index] >= '0') and (Result[Index] <= '9')) or
      (Result[Index] = '.')
    ) then
    begin
      Valid := False;
      Exit;
    end;
    if (
      (Result[Index] = '.') and
      (Index < Length(Result)) and
      (Result[Index + 1] = '.')
    ) then
    begin
      Valid := False;
      Exit;
    end;
  end;
end;

function ReadVersionPart(
  const Value: String;
  var Position: Integer;
  var Valid: Boolean
): Integer;
var
  StartPosition: Integer;
  Token: String;
begin
  Result := 0;
  if Position > Length(Value) then
    Exit;
  StartPosition := Position;
  while (
    (Position <= Length(Value)) and
    (Value[Position] >= '0') and
    (Value[Position] <= '9')
  ) do
    Position := Position + 1;
  if StartPosition = Position then
  begin
    Valid := False;
    Exit;
  end;
  Token := Copy(Value, StartPosition, Position - StartPosition);
  if (Length(Token) > 1) and (Token[1] = '0') then
  begin
    Valid := False;
    Exit;
  end;
  Result := StrToIntDef(Token, -1);
  if Result < 0 then
  begin
    Valid := False;
    Exit;
  end;
  if Position <= Length(Value) then
  begin
    if Value[Position] <> '.' then
    begin
      Valid := False;
      Exit;
    end;
    Position := Position + 1;
  end;
end;

function CompareRuntimeVersions(
  const LeftValue: String;
  const RightValue: String;
  var Valid: Boolean
): Integer;
var
  Left: String;
  Right: String;
  LeftPosition: Integer;
  RightPosition: Integer;
  LeftPart: Integer;
  RightPart: Integer;
begin
  Result := 0;
  Valid := True;
  Left := NormalizeRuntimeVersion(LeftValue, Valid);
  if not Valid then
    Exit;
  Right := NormalizeRuntimeVersion(RightValue, Valid);
  if not Valid then
    Exit;
  LeftPosition := 1;
  RightPosition := 1;
  while (
    (LeftPosition <= Length(Left)) or
    (RightPosition <= Length(Right))
  ) do
  begin
    LeftPart := ReadVersionPart(Left, LeftPosition, Valid);
    if not Valid then
      Exit;
    RightPart := ReadVersionPart(Right, RightPosition, Valid);
    if not Valid then
      Exit;
    if LeftPart < RightPart then
    begin
      Result := -1;
      Exit;
    end;
    if LeftPart > RightPart then
    begin
      Result := 1;
      Exit;
    end;
  end;
end;

function CheckNoDowngrade(
  MarkerName: String;
  DisplayName: String;
  var ErrorMessage: String
): Boolean;
var
  InstalledVersion: String;
  Valid: Boolean;
  Comparison: Integer;
begin
  Result := True;
  InstalledVersion := LoadMarker(MarkerName);
  if InstalledVersion = '' then
    Exit;
  Comparison := CompareRuntimeVersions(
    '{#MyAppVersion}', InstalledVersion, Valid
  );
  if not Valid then
  begin
    ErrorMessage :=
      'The installed ' + DisplayName +
      ' runtime marker is invalid, so safe numeric ordering cannot be ' +
      'proved. Restore the marker from a known backup or contact support.';
    Result := False;
    Exit;
  end;
  if Comparison < 0 then
  begin
    ErrorMessage :=
      'Update {#MyAppVersion} is older than the installed ' +
      DisplayName + ' runtime ' + InstalledVersion +
      '. Downgrades are blocked to protect customer data.';
    Result := False;
  end;
end;

function VerifyRollbackCandidate(Version: String): Boolean;
var
  SelfTestData: String;
  SelfTestOutput: String;
  SelfTestParameters: String;
  ResultCode: Integer;
begin
  Result := False;
  Version := Trim(Version);
  if (Version = '') or
     (CompareText(Version, '{#MyAppVersion}') = 0) or
     not FileExists(
       RuntimeRoot() + '\' + Version + '\runtime-manifest.json'
     ) then
    Exit;
  SelfTestData := ExpandConstant('{tmp}\bcvision-rollback-self-test');
  SelfTestOutput := ExpandConstant(
    '{tmp}\bcvision-rollback-self-test.json'
  );
  DelTree(SelfTestData, True, True, True);
  DeleteFile(SelfTestOutput);
  ForceDirectories(SelfTestData);
  SelfTestParameters :=
    '--self-test --runtime-candidate ' + AddQuotes(Version) +
    ' --self-test-data-dir ' + AddQuotes(SelfTestData) +
    ' --self-test-output ' + AddQuotes(SelfTestOutput);
  ResultCode := -1;
  Result := Exec(
    ExpandConstant('{app}\{#MyAppExeName}'),
    SelfTestParameters,
    ExpandConstant('{app}'),
    SW_HIDE,
    ewWaitUntilTerminated,
    ResultCode
  ) and (ResultCode = 0);
end;

function TryRollbackCandidate(
  Version: String;
  var SelectedVersion: String
): Boolean;
begin
  Result := False;
  Version := Trim(Version);
  if (Version = '') or
     (CompareText(Version, SelectedVersion) = 0) then
    Exit;
  if VerifyRollbackCandidate(Version) then
  begin
    SelectedVersion := Version;
    Result := True;
  end;
end;

function FindVerifiedRollback(): String;
begin
  Result := '';
  if TryRollbackCandidate(LoadMarker('current.txt'), Result) then
    Exit;
  if TryRollbackCandidate(LoadMarker('last-known-good.txt'), Result) then
    Exit;
  if TryRollbackCandidate(LoadMarker('previous.txt'), Result) then
    Exit;
  TryRollbackCandidate('{#MyBaseVersion}', Result);
end;

function RestoreFailedUpdate(RollbackVersion: String): Boolean;
var
  CurrentRestored: Boolean;
  LastGoodRestored: Boolean;
  FailureRecorded: Boolean;
begin
  if (RollbackVersion = '') or
     (CompareText(RollbackVersion, '{#MyAppVersion}') = 0) then
  begin
    Result := False;
    Exit;
  end;
  CurrentRestored := AtomicWriteMarker(
    'current.txt', RollbackVersion
  );
  LastGoodRestored := AtomicWriteMarker(
    'last-known-good.txt', RollbackVersion
  );
  FailureRecorded := AtomicWriteMarker(
    'failed.txt', '{#MyAppVersion}'
  );
  Result := CurrentRestored and LastGoodRestored and FailureRecorded;
  if Result then
    DeleteMarker('pending.txt');
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  RuntimeAbi: AnsiString;
  RuntimeContract: AnsiString;
  ResultCode: Integer;
begin
  Result := '';
  if not FileExists(ExpandConstant('{app}\BCVision.exe')) then
  begin
    Result := 'A compatible BC Vision full base is not installed. Run the full installer once.';
    Exit;
  end;
  if not LoadStringFromFile(
    ExpandConstant('{app}\runtime-abi.txt'), RuntimeAbi
  ) then
  begin
    Result := 'This installation does not support fast updates. Install the current full base once.';
    Exit;
  end;
  if Trim(RuntimeAbi) <> '{#MyRuntimeAbi}' then
  begin
    Result := 'This update needs a newer full BC Vision runtime.';
    Exit;
  end;
  if not LoadStringFromFile(
    ExpandConstant('{app}\runtime-contract.txt'), RuntimeContract
  ) or (CompareText(Trim(RuntimeContract), '{#MyRuntimeContract}') <> 0) then
  begin
    Result := 'The installed BC Vision runtime does not match this update.';
    Exit;
  end;

  if not CheckNoDowngrade(
    'current.txt', 'current', Result
  ) then
    Exit;
  if not CheckNoDowngrade(
    'last-known-good.txt', 'last-known-good', Result
  ) then
    Exit;

  Exec(
    ExpandConstant('{cmd}'),
    '/C taskkill /IM BCVision.exe /T /F >nul 2>&1',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode
  );

  RollbackVersion := FindVerifiedRollback();
  if RollbackVersion = '' then
  begin
    Result :=
      'No different verified runtime is available for rollback. ' +
      'Run the full repair installer before reinstalling this update.';
    Exit;
  end;
  if not AtomicWriteMarker('previous.txt', RollbackVersion) then
  begin
    Result :=
      'The verified rollback pointer could not be saved before staging.';
    Exit;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  SelfTestData: String;
  SelfTestOutput: String;
  SelfTestParameters: String;
  ResultCode: Integer;
  SelfTestStarted: Boolean;
begin
  if CurStep <> ssPostInstall then
    Exit;

  if (RollbackVersion = '') or
     (CompareText(RollbackVersion, '{#MyAppVersion}') = 0) then
    RaiseException(
      'No different verified BC Vision runtime is available for rollback.'
    );

  if not AtomicWriteMarker('pending.txt', '{#MyAppVersion}') then
    RaiseException('The update transaction could not be staged.');
  if not AtomicWriteMarker('current.txt', '{#MyAppVersion}') then
  begin
    DeleteMarker('pending.txt');
    RaiseException('The update candidate could not be selected.');
  end;

  SelfTestData := ExpandConstant('{tmp}\bcvision-update-self-test');
  SelfTestOutput := ExpandConstant('{tmp}\bcvision-update-self-test.json');
  DelTree(SelfTestData, True, True, True);
  DeleteFile(SelfTestOutput);
  ForceDirectories(SelfTestData);
  SelfTestParameters :=
    '--self-test --runtime-candidate ' + AddQuotes('{#MyAppVersion}') +
    ' --self-test-data-dir ' + AddQuotes(SelfTestData) +
    ' --self-test-output ' + AddQuotes(SelfTestOutput);
  ResultCode := -1;
  SelfTestStarted := Exec(
    ExpandConstant('{app}\{#MyAppExeName}'),
    SelfTestParameters,
    ExpandConstant('{app}'),
    SW_HIDE,
    ewWaitUntilTerminated,
    ResultCode
  );

  if (not SelfTestStarted) or (ResultCode <> 0) then
  begin
    RestoreFailedUpdate(RollbackVersion);
    RaiseException(
      'The new runtime failed its installed self-test. ' +
      'BC Vision restored the previous version or will recover it ' +
      'automatically on the next launch.'
    );
  end;

  if not AtomicWriteMarker('current.txt', '{#MyAppVersion}') then
  begin
    RestoreFailedUpdate(RollbackVersion);
    RaiseException(
      'The verified runtime could not be selected. ' +
      'BC Vision will use the previous version.'
    );
  end;

  if not AtomicWriteMarker(
    'last-known-good.txt',
    '{#MyAppVersion}'
  ) then
  begin
    RestoreFailedUpdate(RollbackVersion);
    RaiseException(
      'The verified runtime could not be committed. ' +
      'BC Vision will use the previous version.'
    );
  end;

  DeleteMarker('pending.txt');
  DeleteMarker('failed.txt');
end;
