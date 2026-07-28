[Setup]
; App Metadata
AppName={#APP_NAME}
AppVersion={#APP_VERSION}
AppPublisher=Revoice
AppId={#APP_ID}
DefaultDirName={localappdata}\Programs\{#APP_ID}
DefaultGroupName={#APP_NAME}
LicenseFile=installer_disclaimer_zh-cn.txt

; Installer Output
OutputDir={#OUT_DIR}
OutputBaseFilename={#OUT_FILENAME}
SetupIconFile={#INSTALLER_ICON}
UninstallDisplayIcon={app}\{#APP_ICON_FILE}

; Compression Settings (LZMA2 Ultra for maximum compression of large AI models)
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMADictionarySize=262144
LZMANumBlockThreads=6

; Current installer payload stays below 4GB, so disk spanning is not needed.
DiskSpanning=no

; Architecture & Privileges
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest
DisableDirPage=no

; UI Settings
DisableWelcomePage=no
ShowLanguageDialog=no
WizardStyle=modern

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The main application files
Source: "{#PACKAGE_DIR}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Note: Don't use "ignoreversion" on any shared system files

[Icons]
Name: "{group}\{#APP_NAME}"; Filename: "{app}\{#APP_LAUNCHER}"; WorkingDir: "{app}"; IconFilename: "{app}\{#APP_ICON_FILE}"
Name: "{group}\{cm:UninstallProgram,{#APP_NAME}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#APP_NAME}"; Filename: "{app}\{#APP_LAUNCHER}"; WorkingDir: "{app}"; IconFilename: "{app}\{#APP_ICON_FILE}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#APP_LAUNCHER}"; Description: "{cm:LaunchProgram,{#StringChange(APP_NAME, '&', '&&')}}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[Dirs]
; Pre-create necessary directories for runtime
Name: "{app}\outputs"
Name: "{app}\checkpoints"
Name: "{app}\logs"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\checkpoints"
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\vendor"
Type: files; Name: "{app}\settings.json"
