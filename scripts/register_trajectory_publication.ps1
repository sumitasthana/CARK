param(
    [string]$Repository = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$publicationRoot = (Resolve-Path -LiteralPath $Repository).Path
$publicationScript = Join-Path $publicationRoot 'scripts/publish_experiment_trajectory.py'
if (-not (Test-Path -LiteralPath $publicationScript -PathType Leaf)) {
    throw 'The experiment publisher script is missing.'
}
if ((Get-TimeZone).Id -ne 'Eastern Standard Time') {
    throw 'This schedule requires Windows to use the New York time zone (Eastern Standard Time).'
}
$publicationPython = (Get-Command python.exe -ErrorAction Stop).Source
$publicationPythonWindowless = Join-Path (Split-Path -Parent $publicationPython) 'pythonw.exe'
if (-not (Test-Path -LiteralPath $publicationPythonWindowless -PathType Leaf)) {
    throw 'pythonw.exe is required to run without opening a console window.'
}
$publicationUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$publicationAction = New-ScheduledTaskAction -Execute $publicationPythonWindowless `
    -Argument ('"' + $publicationScript + '" --publish') -WorkingDirectory $publicationRoot
$publicationTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Wednesday,Saturday -At '18:00'
$publicationPrincipal = New-ScheduledTaskPrincipal -UserId $publicationUser -LogonType Interactive -RunLevel Limited
$publicationSettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 20) -WakeToRun `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName 'CARK-Experiment-Trajectory-Publish' `
    -Description 'Publish the committed experiment trajectory on Wednesday and Saturday at 6 p.m. New York time.' `
    -Action $publicationAction -Trigger $publicationTrigger -Principal $publicationPrincipal `
    -Settings $publicationSettings -Force | Select-Object TaskName,State
Get-ScheduledTaskInfo -TaskName 'CARK-Experiment-Trajectory-Publish' | Select-Object NextRunTime
