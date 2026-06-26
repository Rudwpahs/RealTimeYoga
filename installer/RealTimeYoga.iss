#define AppName "RealTimeYoga"
#ifndef AppVersion
#define AppVersion "dev"
#endif
#ifndef SourceDir
#define SourceDir "..\dist\RealTimeYoga"
#endif
#ifndef OutputDir
#define OutputDir "..\dist"
#endif

[Setup]
AppId={{7A8BB51B-3B9E-49F4-8848-AB5667EAC695}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Rudwpahs
AppPublisherURL=https://github.com/Rudwpahs/RealTimeYoga
AppSupportURL=https://github.com/Rudwpahs/RealTimeYoga/issues
AppUpdatesURL=https://github.com/Rudwpahs/RealTimeYoga/releases
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename={#AppName}-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\RealTimeYoga.exe
PrivilegesRequired=lowest
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\RealTimeYoga"; Filename: "{app}\RealTimeYoga.exe"
Name: "{group}\Uninstall RealTimeYoga"; Filename: "{uninstallexe}"
Name: "{autodesktop}\RealTimeYoga"; Filename: "{app}\RealTimeYoga.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\RealTimeYoga.exe"; Description: "{cm:LaunchProgram,RealTimeYoga}"; Flags: nowait postinstall skipifsilent
