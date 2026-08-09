const
  RuntimeMoveFileReplaceExisting = 1;
  RuntimeMoveFileWriteThrough = 8;

var
  RuntimeVersionToActivate: String;
  InvalidRuntimeMarkerFound: Boolean;

function RuntimeMoveFileEx(
  ExistingFileName: String;
  NewFileName: String;
  Flags: Cardinal
): Boolean;
  external 'MoveFileExW@kernel32.dll stdcall setuponly';

function RuntimeRoot(): String;
begin
  Result := ExpandConstant('{app}\runtime');
end;

function RuntimeMarkerPath(Name: String): String;
begin
  Result := RuntimeRoot() + '\' + Name;
end;

function LoadRuntimeMarker(Name: String): String;
var
  Value: AnsiString;
begin
  Result := '';
  if LoadStringFromFile(RuntimeMarkerPath(Name), Value) then
    Result := Trim(Value);
end;

function AtomicWriteRuntimeMarker(Name: String; Value: String): Boolean;
var
  Destination: String;
  Temporary: String;
begin
  ForceDirectories(RuntimeRoot());
  Destination := RuntimeMarkerPath(Name);
  Temporary := RuntimeMarkerPath('.' + Name + '.full.tmp');
  DeleteFile(Temporary);
  Result := SaveStringToFile(Temporary, Value + #13#10, False);
  if Result then
    Result := RuntimeMoveFileEx(
      Temporary,
      Destination,
      RuntimeMoveFileReplaceExisting or RuntimeMoveFileWriteThrough
    );
  if not Result then
    DeleteFile(Temporary);
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

function ReadRuntimeVersionPart(
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
    Position := Position + 1;
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
    LeftPart := ReadRuntimeVersionPart(Left, LeftPosition, Valid);
    if not Valid then
      Exit;
    RightPart := ReadRuntimeVersionPart(Right, RightPosition, Valid);
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

function IsCompatibleNewerRuntime(
  Version: String;
  var Valid: Boolean
): Boolean;
var
  Comparison: Integer;
  Prefix: String;
begin
  Comparison := CompareRuntimeVersions(Version, '{#MyAppVersion}', Valid);
  Prefix := '{#MyAppVersion}.';
  Result := Valid and (Comparison > 0) and
    (CompareText(Copy(Version, 1, Length(Prefix)), Prefix) = 0);
end;

function VerifyInstalledRuntimeCandidate(Version: String): Boolean;
var
  SelfTestData: String;
  SelfTestOutput: String;
  Parameters: String;
  ResultCode: Integer;
begin
  SelfTestData := ExpandConstant('{tmp}\bcvision-full-repair-self-test');
  SelfTestOutput := ExpandConstant(
    '{tmp}\bcvision-full-repair-self-test.json'
  );
  DelTree(SelfTestData, True, True, True);
  DeleteFile(SelfTestOutput);
  ForceDirectories(SelfTestData);
  Parameters :=
    '--self-test --runtime-candidate ' + AddQuotes(Version) +
    ' --self-test-data-dir ' + AddQuotes(SelfTestData) +
    ' --self-test-output ' + AddQuotes(SelfTestOutput);
  ResultCode := -1;
  Result := Exec(
    ExpandConstant('{app}\{#MyAppExeName}'),
    Parameters,
    ExpandConstant('{app}'),
    SW_HIDE,
    ewWaitUntilTerminated,
    ResultCode
  ) and (ResultCode = 0);
end;

function ConsiderNewerRuntime(
  Version: String;
  var Selected: String;
  var ErrorMessage: String
): Boolean;
var
  Valid: Boolean;
  Comparison: Integer;
begin
  Result := True;
  Version := Trim(Version);
  if Version = '' then
    Exit;
  Comparison := CompareRuntimeVersions(Version, '{#MyAppVersion}', Valid);
  if not Valid then
  begin
    InvalidRuntimeMarkerFound := True;
    Exit;
  end;
  if Comparison <= 0 then
    Exit;
  if not IsCompatibleNewerRuntime(Version, Valid) then
  begin
    ErrorMessage :=
      'Installed runtime ' + Version +
      ' is newer but is not compatible with the RC29 base installer. ' +
      'Downgrade was refused.';
    Result := False;
    Exit;
  end;
  if Selected = '' then
    Selected := Version
  else
  begin
    Comparison := CompareRuntimeVersions(Version, Selected, Valid);
    if Valid and (Comparison > 0) then
      Selected := Version;
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  CurrentVersion: String;
  LastGoodVersion: String;
  PendingVersion: String;
  SelectedNewer: String;
  ResultCode: Integer;
begin
  Result := '';
  RuntimeVersionToActivate := '{#MyAppVersion}';
  InvalidRuntimeMarkerFound := False;
#if RequireExistingInstall
  if not FileExists(ExpandConstant('{app}\BCVision.exe')) then
  begin
    Result := 'BC Vision is not installed. Run the full installer first.';
    Exit;
  end;
#endif
  if not FileExists(ExpandConstant('{app}\BCVision.exe')) then
    Exit;

  Exec(
    ExpandConstant('{cmd}'),
    '/C taskkill /IM BCVision.exe /T /F >nul 2>&1',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode
  );
  CurrentVersion := LoadRuntimeMarker('current.txt');
  LastGoodVersion := LoadRuntimeMarker('last-known-good.txt');
  PendingVersion := LoadRuntimeMarker('pending.txt');

  { An unconfirmed pending current is not a last-known-good version. }
  if (PendingVersion <> '') and
     not (
       (CompareText(PendingVersion, CurrentVersion) = 0) and
       (CompareText(PendingVersion, LastGoodVersion) = 0)
     ) then
  begin
    if CompareText(PendingVersion, CurrentVersion) = 0 then
      CurrentVersion := '';
    if CompareText(PendingVersion, LastGoodVersion) = 0 then
      LastGoodVersion := '';
  end;

  SelectedNewer := '';
  if not ConsiderNewerRuntime(
    CurrentVersion, SelectedNewer, Result
  ) then
    Exit;
  if not ConsiderNewerRuntime(
    LastGoodVersion, SelectedNewer, Result
  ) then
    Exit;
  if SelectedNewer = '' then
  begin
    if InvalidRuntimeMarkerFound then
      Result :=
        'Runtime pointers are corrupt and no verified compatible newer ' +
        'runtime can be proven. RC29 repair refused a possible downgrade; ' +
        'restore the pointers from backup or contact support.';
    Exit;
  end;
  if not VerifyInstalledRuntimeCandidate(SelectedNewer) then
  begin
    Result :=
      'Installed newer runtime ' + SelectedNewer +
      ' did not pass the isolated self-test. Full RC29 repair refused ' +
      'to activate an older runtime against newer customer data.';
    Exit;
  end;
  RuntimeVersionToActivate := SelectedNewer;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep <> ssPostInstall then
    Exit;
  if not AtomicWriteRuntimeMarker(
    'pending.txt', RuntimeVersionToActivate
  ) then
    RaiseException('The full-runtime transaction could not be staged.');
  if not AtomicWriteRuntimeMarker(
    'current.txt', RuntimeVersionToActivate
  ) then
    RaiseException(
      'The runtime pointer could not be committed; the pending marker ' +
      'was retained for safe launcher fallback.'
    );
  if not AtomicWriteRuntimeMarker(
    'last-known-good.txt', RuntimeVersionToActivate
  ) then
    RaiseException(
      'The last-known-good runtime could not be committed; the pending ' +
      'marker was retained for safe launcher fallback.'
    );
  DeleteFile(RuntimeMarkerPath('pending.txt'));
  DeleteFile(RuntimeMarkerPath('failed.txt'));
end;
