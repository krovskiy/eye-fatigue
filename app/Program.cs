using System.Diagnostics;
using System.Drawing;
using System.Windows.Forms;

namespace EyeFatigue;

internal static class Program
{
    [STAThread]
    static void Main(string[] args)
    {
        if (args.Contains("--self-test")) { SelfTest.Run(); return; }
        ApplicationConfiguration.Initialize();
        Application.Run(new MainForm());
    }
}

internal sealed class BreakClock
{
    private readonly Stopwatch watch = new();
    public int IntervalSeconds { get; private set; } = 20 * 60;
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
    }
}

internal sealed class MainForm : Form
{
    readonly Color background = Color.FromArgb(244, 248, 247);
    readonly Color ink = Color.FromArgb(26, 53, 50);
    readonly Color accent = Color.FromArgb(22, 109, 94);
    readonly BreakClock clock = new();
    readonly System.Windows.Forms.Timer timer = new() { Interval = 500 };
    readonly NotifyIcon tray = new() { Icon = SystemIcons.Information, Text = "Eye fatigue reminders", Visible = true };
    readonly Label status = new(), time = new(), detail = new(), audit = new();
    readonly ProgressBar progress = new();
    readonly Button toggle = new(), reset = new(), tired = new(), done = new(), history = new();
    readonly NumericUpDown interval = new() { Minimum = 5, Maximum = 120, Value = 20, Width = 76, Font = new Font("Segoe UI", 12) };
    bool notified;
    readonly string auditPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "EyeFatigue", "activity.csv");

    public MainForm()
    {
        Text = "Eye fatigue · break reminders"; MinimumSize = new Size(620, 650); Size = new Size(760, 760);
        StartPosition = FormStartPosition.CenterScreen; BackColor = background; ForeColor = ink; Font = new Font("Segoe UI", 11);
        var root = new TableLayoutPanel { Dock = DockStyle.Fill, Padding = new Padding(28), ColumnCount = 1, AutoScroll = true };
        Controls.Add(root);
        AddLabel(root, "EYE FATIGUE", 11, FontStyle.Bold, accent);
        AddLabel(root, "Give your eyes a break.", 25, FontStyle.Bold, ink);
        AddLabel(root, "A small reminder to look away from your screen and check how you feel.", 11, FontStyle.Regular, ink);
        status.Text = "Ready when you are"; status.Font = new Font("Segoe UI", 15, FontStyle.Bold); status.AutoSize = true; status.Margin = new Padding(0, 28, 0, 4); root.Controls.Add(status);
        time.Font = new Font("Segoe UI", 34, FontStyle.Bold); time.AutoSize = true; time.Margin = new Padding(0, 0, 0, 8); root.Controls.Add(time);
        progress.Dock = DockStyle.Top; progress.Height = 20; progress.Maximum = 1000; progress.AccessibleName = "Time until your next break"; root.Controls.Add(progress);
        detail.Text = "Every 20 minutes, look about 6 metres (20 feet) away for 20 seconds."; detail.AutoSize = true; detail.MaximumSize = new Size(620, 0); detail.Margin = new Padding(0, 12, 0, 16); root.Controls.Add(detail);
        var buttons = new FlowLayoutPanel { AutoSize = true, Dock = DockStyle.Top, WrapContents = true }; root.Controls.Add(buttons);
        SetupButton(toggle, "Start", true); SetupButton(reset, "Reset", false); SetupButton(tired, "My eyes feel tired", false); SetupButton(done, "Break done", false);
        SetupButton(history, "Open activity", false); buttons.Controls.AddRange([toggle, reset, tired, done, history]);
        var settings = new FlowLayoutPanel { AutoSize = true, Dock = DockStyle.Top, Margin = new Padding(0, 20, 0, 0) }; root.Controls.Add(settings);
        settings.Controls.Add(new Label { Text = "Remind me every", AutoSize = true, Margin = new Padding(0, 7, 8, 0) }); settings.Controls.Add(interval);
        settings.Controls.Add(new Label { Text = "minutes", AutoSize = true, Margin = new Padding(8, 7, 0, 0) });
        AddLabel(root, "YOUR ACTIVITY", 11, FontStyle.Bold, accent, 24);
        audit.AutoSize = true; audit.MaximumSize = new Size(620, 0); audit.Text = "No activity yet. Your activity stays on this device."; root.Controls.Add(audit);
        AddLabel(root, "This is a wellness reminder, not a medical device. It does not detect or diagnose eye fatigue. If symptoms persist or worry you, speak with an eye care professional.", 10, FontStyle.Regular, ink, 22);
        toggle.Click += (_, _) => { if (clock.Running) { clock.Pause(); Log("paused"); } else { if (clock.Due) { clock.Reset((int)interval.Value); notified = false; } clock.Start(); Log("started"); } RefreshView(); };
        reset.Click += (_, _) => { clock.Reset((int)interval.Value); notified = false; Log("reset"); RefreshView(); };
        done.Click += (_, _) => { clock.Reset((int)interval.Value); clock.Start(); notified = false; Log("break marked done"); RefreshView(); };
        tired.Click += (_, _) => { clock.Pause(); Log("eyes feel tired"); Notify("Time for a break", "Look away from your screen and rest your eyes. Take more time if you need it."); RefreshView(); };
        interval.ValueChanged += (_, _) => { clock.Reset((int)interval.Value); notified = false; Log("interval changed to " + interval.Value + " minutes"); RefreshView(); };
        history.Click += (_, _) => { try { if (!File.Exists(auditPath)) { MessageBox.Show("Your activity will appear here after you use the timer.", "Activity"); return; } Process.Start(new ProcessStartInfo(auditPath) { UseShellExecute = true }); } catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or IOException or UnauthorizedAccessException) { MessageBox.Show("Could not open activity: " + ex.Message, "Activity"); } };
        timer.Tick += (_, _) => { if (clock.Due && !notified) { clock.Pause(); notified = true; Log("break reminder"); Notify("Time for a break", "Look about 6 metres away for 20 seconds, then mark your break done."); } RefreshView(); };
        tray.DoubleClick += (_, _) => { Show(); WindowState = FormWindowState.Normal; Activate(); };
        FormClosed += (_, _) => { timer.Stop(); tray.Dispose(); };
        timer.Start(); RefreshView();
    }
    void AddLabel(Control parent, string text, float size, FontStyle style, Color color, int top = 0)
    { var label = new Label { Text = text, Font = new Font("Segoe UI", size, style), ForeColor = color, AutoSize = true, MaximumSize = new Size(620, 0), Margin = new Padding(0, top, 0, 6) }; parent.Controls.Add(label); }
    void SetupButton(Button button, string text, bool primary)
    { button.Text = text; button.AutoSize = true; button.MinimumSize = new Size(110, 44); button.Margin = new Padding(0, 0, 8, 8); button.FlatStyle = FlatStyle.Flat; button.BackColor = primary ? accent : Color.White; button.ForeColor = primary ? Color.White : ink; button.UseVisualStyleBackColor = false; }
    void RefreshView()
    {
        int left = clock.Remaining; time.Text = $"{left / 60:00}:{left % 60:00}";
        progress.Value = Math.Clamp((clock.IntervalSeconds - left) * 1000 / clock.IntervalSeconds, 0, 1000);
        status.Text = clock.Due ? "Time to look away" : clock.Running ? "Your next break in" : "Paused · ready when you are";
        toggle.Text = clock.Running ? "Pause" : "Start";
    }
    void Notify(string title, string message) { tray.BalloonTipTitle = title; tray.BalloonTipText = message; tray.ShowBalloonTip(10000); }
    void Log(string action)
    {
        audit.Text = DateTime.Now.ToString("g") + "  ·  " + action;
        try { Directory.CreateDirectory(Path.GetDirectoryName(auditPath)!); if (!File.Exists(auditPath)) File.AppendAllText(auditPath, "time,action\n"); File.AppendAllText(auditPath, DateTimeOffset.Now.ToString("O") + "," + action + "\n"); }
        catch (IOException) { audit.Text += "  ·  could not save activity"; }
        catch (UnauthorizedAccessException) { audit.Text += "  ·  could not save activity"; }
    }
}

