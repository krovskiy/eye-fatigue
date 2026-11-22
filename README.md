# eye fatigue

A small Windows app to help you remember to take breaks from the screen. Start the timer, look away when it reminds you, and mark your break done when you are ready. You can also pause it or tell it your eyes feel tired.

The default is the 20-20-20 reminder: every 20 minutes, look about 20 feet (6 metres) away for 20 seconds. You can change the interval from 5 to 120 minutes. The progress bar shows time to the next reminder, not a measurement of your eyes.

## get started

You need Windows and the [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0). In PowerShell, from this folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run.ps1
```

Press **Start** to begin. Windows may hide notifications when Do Not Disturb is on or notifications are disabled. Keep the app open for reminders; it does not run in the background after you close it. **Break done** starts the next interval. Changing the interval or resetting pauses the timer, so press Start again.

Build with `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build.ps1` or run the clock checks with `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\test.ps1`. GitHub Actions builds and runs those checks on Windows for pushes and pull requests.

## activity and privacy

The latest action is shown in the app. Use **Open activity** to open the full history. A simple activity history is saved locally at `%LOCALAPPDATA%\EyeFatigue\activity.csv`; open it in a spreadsheet or text editor, and delete the file whenever you want. It contains timestamps and actions such as reminders, pauses and self-reported tired eyes. It does not contain images. No account or network connection is needed by the app. It does not currently use the camera.

## what this can and cannot do

This is a wellness reminder, **not a medical device**, diagnosis, or a validated fatigue detector. The timer does not know whether you are looking at the screen or whether you took a break. The original OpenCV experiment in `eye-fatigue/` is separate from the app: detecting a face or failing to find an eye is not enough to determine eye closure or fatigue. It is not used to make health claims or trigger reminders. Camera-based detection needs validated landmarks, signal-quality checks, careful evaluation across users and conditions, and clinical review before it could support such claims.

If symptoms persist or concern you, speak with an eye care professional. For the evidence behind these choices and its limits, see [research.md](research.md).

