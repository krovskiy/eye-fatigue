using System.Diagnostics;

namespace EyeFatigue;

internal sealed class MainForm : Form
{
    readonly BreakClock clock = new();
    readonly Store store;
    readonly Settings settings;
    readonly bool demo;
    readonly System.Windows.Forms.Timer timer = new() { Interval = 500 };
    readonly NotifyIcon tray = new() { Icon = SystemIcons.Information, Text = "Eye fatigue reminders", Visible = true };
    readonly SplitContainer split = new() { Size = new Size(1000, 700), Dock = DockStyle.Fill, SplitterWidth = 7, Panel1MinSize = 280, Panel2MinSize = 400 };
    readonly Panel pages = new() { Dock = DockStyle.Fill };
    readonly FlowLayoutPanel navigation = new() { Dock = DockStyle.Top, Height = 64, Padding = new Padding(20, 10, 0, 0), WrapContents = false };
    readonly List<Panel> pageList = [];
    readonly List<SoftButton> navButtons = [];
    int selectedPage;
    readonly Label status = new(), time = new(), detail = new(), notice = new(), stats = new(), latest = new(), breakLegend = Label("■  Breaks marked done", 10, true), tiredLegend = Label("■  Eyes felt tired", 10, true);
    readonly SoftProgress progress = new();
    readonly SoftButton toggle = new(), reset = new(), tired = new(), done = new(), history = new();
    readonly NumericUpDown interval = new() { Minimum = 5, Maximum = 120, Width = 90 };
    readonly CheckBox dark = new() { Text = "Dark theme", AutoSize = true }, notifications = new() { Text = "Windows notifications", AutoSize = true };
    readonly ActivityGraph graph = new() { Dock = DockStyle.Fill };
    readonly ListView list = new() { Dock = DockStyle.Fill, View = View.Details, FullRowSelect = true, HeaderStyle = ColumnHeaderStyle.Nonclickable, AccessibleName = "Recent activity" };
    readonly List<Activity> entries;
    bool notified, loading = true;
    Color bg, panel, fg, muted, accent;

    public MainForm(bool demo = false)
    {
        this.demo = demo;
        store = new Store(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "EyeFatigue"));
        settings = demo ? new Settings() : store.LoadSettings();
        entries = demo ? DemoEntries() : store.ReadActivity();
        Text = "eye fatigue"; MinimumSize = new Size(850, 650); Size = new Size(1120, 760); StartPosition = FormStartPosition.CenterScreen;
        Font = new Font("Segoe UI", 10); KeyPreview = true;
        interval.Value = settings.Interval; dark.Checked = settings.Dark; notifications.Checked = settings.Notifications;
        clock.Reset(settings.Interval);
        Controls.Add(split); split.SplitterDistance = Math.Clamp(settings.PaneWidth, 280, 500);
        BuildTimer(); BuildTabs(); ApplyTheme(); UpdateActivity(); RefreshView();
        toggle.Click += (_, _) => Toggle(); reset.Click += (_, _) => ResetClock();
        done.Click += (_, _) => { clock.Reset(settings.Interval); clock.Start(); notified = false; Log("break marked done"); RefreshView(); };
        tired.Click += (_, _) => { clock.Pause(); Log("eyes feel tired"); Notify("Time for a break", "Look away from your screen and rest your eyes. Take more time if you need it."); RefreshView(); };
        history.Click += (_, _) => OpenHistory();
        interval.ValueChanged += (_, _) => { if (loading) return; settings.Interval = (int)interval.Value; clock.Reset(settings.Interval); notified = false; SaveSettings(); RefreshView(); };
        dark.CheckedChanged += (_, _) => { if (loading) return; settings.Dark = dark.Checked; SaveSettings(); ApplyTheme(); };
        notifications.CheckedChanged += (_, _) => { if (loading) return; settings.Notifications = notifications.Checked; SaveSettings(); };
        split.SplitterMoved += (_, _) => { if (!loading) { settings.PaneWidth = split.SplitterDistance; SaveSettings(); } };
        timer.Tick += (_, _) => { if (clock.Due && !notified) { clock.Pause(); notified = true; Log("break reminder"); Notify("Time for a break", "Look about 6 metres away for 20 seconds, then mark your break done."); } RefreshView(); };
        KeyDown += OnKeyDown;
        tray.DoubleClick += (_, _) => { Show(); WindowState = FormWindowState.Normal; Activate(); };
        FormClosed += (_, _) => { timer.Stop(); tray.Dispose(); };
        loading = false; if (!demo) timer.Start();
    }
    static List<Activity> DemoEntries()
    {
        var result = new List<Activity>(); int[] breaks = [2, 4, 3, 5, 2, 4, 3], reports = [1, 0, 1, 2, 0, 1, 1];
        for (int i = 0; i < 7; i++) { var day = DateTimeOffset.Now.AddDays(i - 6); for (int j = 0; j < breaks[i]; j++) result.Add(new(day.AddMinutes(-j * 30), "break marked done")); for (int j = 0; j < reports[i]; j++) result.Add(new(day.AddMinutes(-j * 35), "eyes feel tired")); }
        return result.OrderByDescending(x => x.Time).ToList();
    }
    static Label Label(string text, float size = 10, bool bold = false) => new() { Text = text, AutoSize = true, Font = new Font("Segoe UI", size, bold ? FontStyle.Bold : FontStyle.Regular), Margin = new Padding(0, 0, 0, 10), MaximumSize = new Size(600, 0) };
    static FlowLayoutPanel Stack(int padding = 24) => new() { Dock = DockStyle.Fill, FlowDirection = FlowDirection.TopDown, WrapContents = false, AutoScroll = true, Padding = new Padding(padding) };
    static void ButtonStyle(Button b, string text) { b.Text = text; b.AutoSize = true; b.MinimumSize = new Size(125, 42); b.FlatStyle = FlatStyle.Flat; b.Margin = new Padding(0, 0, 8, 8); b.UseVisualStyleBackColor = false; }
    void BuildTimer()
    {
        var stack = Stack(); split.Panel1.Controls.Add(stack);
        stack.Controls.Add(Label("◉   EYE FATIGUE", 12, true)); stack.Controls.Add(Label("Your break timer", 20, true));
        stack.Controls.Add(Label("A little space for your eyes.", 10));
        status.AutoSize = true; status.Font = new Font("Segoe UI", 12, FontStyle.Bold); status.Margin = new Padding(0, 25, 0, 3); stack.Controls.Add(status);
        time.AutoSize = true; time.Font = new Font("Segoe UI", 38, FontStyle.Bold); time.Margin = new Padding(0, 0, 0, 10); stack.Controls.Add(time);
        progress.Width = 270; progress.Dock = DockStyle.None; stack.Controls.Add(progress);
        detail.Text = "Look about 6 metres (20 feet) away for 20 seconds."; detail.MaximumSize = new Size(270, 0); detail.AutoSize = true; detail.Margin = new Padding(0, 14, 0, 18); stack.Controls.Add(detail);
        var row = new FlowLayoutPanel { AutoSize = true, Width = 290, WrapContents = true }; stack.Controls.Add(row);
        ButtonStyle(toggle, "Start"); ButtonStyle(reset, "Reset"); ButtonStyle(done, "Break done"); ButtonStyle(tired, "Eyes feel tired"); row.Controls.AddRange([toggle, reset, done, tired]);
        stack.Controls.Add(Label("CTRL+SPACE  start / pause\nCTRL+B  break done\nCTRL+R  reset", 9));
        notice.AutoSize = true; notice.MaximumSize = new Size(270, 0); stack.Controls.Add(notice);
    }
    void BuildTabs()
    {
        split.Panel2.Controls.Add(pages); split.Panel2.Controls.Add(navigation);
        var overview = new Panel { Dock = DockStyle.Fill }; var activity = new Panel { Dock = DockStyle.Fill }; var preferences = new Panel { Dock = DockStyle.Fill };
        pageList.AddRange([overview, activity, preferences]);
        foreach (var page in pageList) pages.Controls.Add(page);
        foreach (var title in new[] { "Overview", "Activity", "Settings" }) { var nav = new SoftButton { Text = title, Width = 110, Height = 40, Margin = new Padding(0, 0, 6, 0), AccessibleName = title }; int index = navButtons.Count; nav.Click += (_, _) => SelectPage(index); navButtons.Add(nav); navigation.Controls.Add(nav); }
        SelectPage(0);
        var grid = new TableLayoutPanel { Dock = DockStyle.Fill, Padding = new Padding(24), ColumnCount = 1, RowCount = 5 };
        grid.RowStyles.Add(new RowStyle(SizeType.AutoSize)); grid.RowStyles.Add(new RowStyle(SizeType.AutoSize)); grid.RowStyles.Add(new RowStyle(SizeType.Percent, 100)); grid.RowStyles.Add(new RowStyle(SizeType.AutoSize)); grid.RowStyles.Add(new RowStyle(SizeType.AutoSize)); overview.Controls.Add(grid);
        grid.Controls.Add(Label("Your week", 22, true)); grid.Controls.Add(Label("Your activity over the last 7 days", 10)); grid.Controls.Add(graph);
        var legend = new FlowLayoutPanel { AutoSize = true, Dock = DockStyle.Fill }; legend.Controls.Add(breakLegend); legend.Controls.Add(tiredLegend); grid.Controls.Add(legend); stats.AutoSize = true; stats.Margin = new Padding(0, 8, 0, 0); grid.Controls.Add(stats);
        var activityGrid = new TableLayoutPanel { Dock = DockStyle.Fill, Padding = new Padding(24), ColumnCount = 1, RowCount = 4 };
        activityGrid.RowStyles.Add(new RowStyle(SizeType.AutoSize)); activityGrid.RowStyles.Add(new RowStyle(SizeType.AutoSize)); activityGrid.RowStyles.Add(new RowStyle(SizeType.Percent, 100)); activityGrid.RowStyles.Add(new RowStyle(SizeType.AutoSize)); activity.Controls.Add(activityGrid);
        activityGrid.Controls.Add(Label("Activity", 22, true)); activityGrid.Controls.Add(Label("Your latest actions, stored on this device.", 10));
        list.Columns.Add("When", 200); list.Columns.Add("Action", 260); activityGrid.Controls.Add(list); ButtonStyle(history, "Open CSV"); activityGrid.Controls.Add(history);
        var opts = Stack(); preferences.Controls.Add(opts); opts.Controls.Add(Label("Make it yours", 22, true)); opts.Controls.Add(Label("Changes are saved on this device.", 10));
        opts.Controls.Add(Label("Remind me every", 11, true)); var row = new FlowLayoutPanel { AutoSize = true, Width = 360 }; row.Controls.Add(interval); row.Controls.Add(Label("minutes", 10)); opts.Controls.Add(row);
        opts.Controls.Add(dark); opts.Controls.Add(notifications); opts.Controls.Add(Label("Changing the interval resets and pauses the timer.", 10));
        opts.Controls.Add(Label("SHORTCUTS", 10, true)); opts.Controls.Add(Label("Ctrl+1  overview    Ctrl+2  activity\nCtrl+3  settings\nCtrl+Space  start / pause\nCtrl+B  break done    Ctrl+R  reset", 10));
        opts.Controls.Add(Label("This is a wellness reminder, not a medical device. It does not detect or diagnose eye fatigue. If symptoms persist or worry you, speak with an eye care professional.", 10));
        latest.Text = ""; latest.AutoSize = true; opts.Controls.Add(latest);
    }
    void ApplyTheme()
    {
        bg = settings.Dark ? Color.FromArgb(35, 36, 40) : Color.FromArgb(235, 236, 241); panel = settings.Dark ? Color.FromArgb(43, 45, 49) : Color.White;
        fg = settings.Dark ? Color.FromArgb(242, 243, 245) : Color.FromArgb(35, 38, 48); muted = settings.Dark ? Color.FromArgb(181, 186, 193) : Color.FromArgb(80, 84, 98); accent = settings.Dark ? Color.FromArgb(155, 163, 255) : Color.FromArgb(79, 86, 170);
        BackColor = bg; ForeColor = fg; ThemeControls(this); split.BackColor = bg; split.Panel1.BackColor = bg; split.Panel2.BackColor = panel;
        graph.Dark = settings.Dark; graph.BackColor = panel; graph.Invalidate(); breakLegend.ForeColor = accent; tiredLegend.ForeColor = settings.Dark ? Color.FromArgb(255, 186, 128) : Color.FromArgb(145, 77, 28); time.ForeColor = accent; detail.ForeColor = muted; notice.ForeColor = muted; latest.ForeColor = muted;
        toggle.BackColor = settings.Dark ? Color.FromArgb(88, 101, 242) : Color.FromArgb(79, 86, 170); toggle.ForeColor = Color.White; progress.Fill = accent; progress.Track = settings.Dark ? Color.FromArgb(65, 67, 74) : Color.FromArgb(220, 222, 231); progress.Invalidate(); RefreshNavigation();
    }
    void ThemeControls(Control c)
    {
        foreach (Control child in c.Controls)
        {
            child.ForeColor = fg;
            if (child is Panel || child is TableLayoutPanel || child is FlowLayoutPanel || child is ListView || child is NumericUpDown || child is ActivityGraph) child.BackColor = panel;
            if (child is Button b) { b.BackColor = settings.Dark ? Color.FromArgb(58, 60, 67) : Color.FromArgb(231, 232, 239); b.ForeColor = fg; b.FlatAppearance.BorderColor = settings.Dark ? Color.FromArgb(70, 100, 98) : Color.FromArgb(180, 200, 194); }
            ThemeControls(child);
        }
        split.Panel1.BackColor = bg;
        foreach (Control c1 in split.Panel1.Controls) { c1.BackColor = bg; foreach (Control c2 in c1.Controls) if (c2 is FlowLayoutPanel) c2.BackColor = bg; }
    }
    void Toggle() { if (clock.Running) { clock.Pause(); Log("paused"); } else { if (clock.Due) { clock.Reset(settings.Interval); notified = false; } clock.Start(); Log("started"); } RefreshView(); }
    void ResetClock() { clock.Reset(settings.Interval); notified = false; Log("reset"); RefreshView(); }
    void RefreshView()
    {
        int left = clock.Remaining; time.Text = $"{left / 60:00}:{left % 60:00}"; progress.AccessibleDescription = $"{left / 60} minutes and {left % 60} seconds until your next break"; progress.Value = Math.Clamp((clock.IntervalSeconds - left) * 1000 / clock.IntervalSeconds, 0, 1000);
        status.Text = clock.Due ? "Time to look away" : clock.Running ? "Next break in" : "Paused · ready when you are"; toggle.Text = clock.Running ? "Pause" : "Start";
    }
    void Log(string action) { if (demo) return; if (store.Add(action)) entries.Insert(0, new Activity(DateTimeOffset.Now, action)); notice.Text = store.Error ?? ""; UpdateActivity(); }
    void UpdateActivity()
    {
        graph.Entries = entries; graph.Invalidate(); var counts = ActivityGraph.Counts(entries, DateTime.Today);
        stats.Text = $"This week   {counts.Sum(x => x.Breaks)} breaks marked done   ·   {counts.Sum(x => x.Tired)} tired-eye reports";
        list.BeginUpdate(); list.Items.Clear(); foreach (var item in entries.Take(100)) list.Items.Add(new ListViewItem([item.Time.LocalDateTime.ToString("g"), item.Action])); list.EndUpdate();
        if (store.Error != null) notice.Text = store.Error;
    }
    void SaveSettings() { if (demo) return; store.SaveSettings(settings); notice.Text = store.Error ?? ""; }
    void Notify(string title, string message) { if (!settings.Notifications) return; tray.BalloonTipTitle = title; tray.BalloonTipText = message; tray.ShowBalloonTip(10000); }
    void OpenHistory() { try { if (!File.Exists(store.ActivityPath)) { MessageBox.Show("Activity will appear after you use the timer.", "Activity"); return; } Process.Start(new ProcessStartInfo(store.ActivityPath) { UseShellExecute = true }); } catch (Exception e) when (e is System.ComponentModel.Win32Exception or IOException or UnauthorizedAccessException) { MessageBox.Show("Could not open activity: " + e.Message, "Activity"); } }
    void OnKeyDown(object? sender, KeyEventArgs e)
    {
        if (!e.Control) return;
        switch (e.KeyCode) { case Keys.D1: SelectPage(0); break; case Keys.D2: SelectPage(1); break; case Keys.D3: SelectPage(2); break; case Keys.Space: Toggle(); break; case Keys.B: done.PerformClick(); break; case Keys.R: ResetClock(); break; default: return; }
        e.Handled = true; e.SuppressKeyPress = true;
    }
    public void SelectPage(int index) { selectedPage = index; for (int i = 0; i < pageList.Count; i++) pageList[i].Visible = i == index; RefreshNavigation(); }
    void RefreshNavigation() { for (int i = 0; i < navButtons.Count; i++) { navButtons[i].BackColor = i == selectedPage ? (settings.Dark ? Color.FromArgb(75, 77, 86) : Color.FromArgb(222, 224, 235)) : panel; navButtons[i].ForeColor = fg; navButtons[i].Invalidate(); } navigation.BackColor = panel; pages.BackColor = panel; }
}
