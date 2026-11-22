using System.Diagnostics;
using System.Text.Json;

namespace EyeFatigue;

internal record Activity(DateTimeOffset Time, string Action);
internal sealed class Settings
{
    public int Interval { get; set; } = 20;
    public bool Dark { get; set; } = true;
    public bool Notifications { get; set; } = true;
    public int PaneWidth { get; set; } = 370;
}
internal sealed class Store
{
    readonly string directory;
    public string ActivityPath => Path.Combine(directory, "activity.csv");
    string SettingsPath => Path.Combine(directory, "settings.json");
    public string? Error { get; private set; }
    public Store(string directory) => this.directory = directory;
    public Settings LoadSettings()
    {
        try { var s = File.Exists(SettingsPath) ? JsonSerializer.Deserialize<Settings>(File.ReadAllText(SettingsPath)) : null; s ??= new(); s.Interval = Math.Clamp(s.Interval, 5, 120); return s; }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or JsonException) { Error = "Could not load settings: " + e.Message; return new(); }
    }
    public void SaveSettings(Settings settings)
    {
        try { Directory.CreateDirectory(directory); File.WriteAllText(SettingsPath, JsonSerializer.Serialize(settings, new JsonSerializerOptions { WriteIndented = true })); Error = null; }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { Error = "Could not save settings: " + e.Message; }
    }
    public List<Activity> ReadActivity()
    {
        try
        {
            if (!File.Exists(ActivityPath)) return [];
            return File.ReadLines(ActivityPath).Skip(1).Select(line => line.Split(',', 2)).Where(parts => parts.Length == 2 && DateTimeOffset.TryParse(parts[0], out _))
                .Select(parts => new Activity(DateTimeOffset.Parse(parts[0]), parts[1])).OrderByDescending(a => a.Time).ToList();
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { Error = "Could not read activity: " + e.Message; return []; }
    }
    public bool Add(string action)
    {
        try { Directory.CreateDirectory(directory); if (!File.Exists(ActivityPath)) File.AppendAllText(ActivityPath, "time,action\n"); File.AppendAllText(ActivityPath, DateTimeOffset.Now.ToString("O") + "," + action + "\n"); Error = null; return true; }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { Error = "Could not save activity: " + e.Message; return false; }
    }
}
internal sealed class BreakClock
{
    readonly Stopwatch watch = new();
    public int IntervalSeconds { get; private set; } = 1200;
    public bool Running => watch.IsRunning;
    public int Remaining => Math.Max(0, IntervalSeconds - (int)watch.Elapsed.TotalSeconds);
    public bool Due => Remaining == 0;
    public void Start() => watch.Start();
    public void Pause() => watch.Stop();
    public void Reset(int minutes) { IntervalSeconds = minutes * 60; watch.Reset(); }
}
internal static class SelfTest
{
    public static void Run()
    {
        var clock = new BreakClock();
        if (clock.Remaining != 1200 || clock.Due || clock.Running) throw new Exception("initial clock");
        clock.Start(); if (!clock.Running) throw new Exception("start");
        clock.Pause(); if (clock.Running) throw new Exception("pause");
        clock.Reset(10); if (clock.Remaining != 600 || clock.Running) throw new Exception("reset");
        clock.Reset(0); if (!clock.Due || clock.Remaining != 0) throw new Exception("due");
        var activities = new List<Activity> { new(DateTimeOffset.Now, "break marked done"), new(DateTimeOffset.Now, "eyes feel tired") };
        var counts = ActivityGraph.Counts(activities, DateTime.Today);
        if (counts[^1].Breaks != 1 || counts[^1].Tired != 1 || counts[0].Breaks != 0) throw new Exception("activity counts");
    }
}
internal static class Program
{
    [STAThread]
    static void Main(string[] args)
    {
        if (args.Contains("--self-test")) { SelfTest.Run(); return; }
        ApplicationConfiguration.Initialize();
        if (args.Length == 2 && args[0] == "--screenshots") { Screenshot.Capture(args[1]); return; }
        Application.Run(new MainForm());
    }
}

