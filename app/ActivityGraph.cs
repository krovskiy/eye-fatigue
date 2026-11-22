using System.Drawing.Drawing2D;

namespace EyeFatigue;

internal sealed class ActivityGraph : Control
{
    public IReadOnlyList<Activity> Entries = [];
    public bool Dark;
    public ActivityGraph() { DoubleBuffered = true; AccessibleName = "Seven day activity graph"; MinimumSize = new Size(300, 210); }
    public static (int Breaks, int Tired)[] Counts(IEnumerable<Activity> entries, DateTime today)
    {
        var days = Enumerable.Range(0, 7).Select(i => today.Date.AddDays(i - 6)).ToArray();
        return days.Select(day => (entries.Count(x => x.Time.LocalDateTime.Date == day && x.Action == "break marked done"), entries.Count(x => x.Time.LocalDateTime.Date == day && x.Action == "eyes feel tired"))).ToArray();
    }
    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e); var g = e.Graphics; g.SmoothingMode = SmoothingMode.AntiAlias;
        var text = Dark ? Color.FromArgb(181, 186, 193) : Color.FromArgb(80, 84, 98);
        var grid = Dark ? Color.FromArgb(67, 69, 76) : Color.FromArgb(224, 226, 233);
        var values = Counts(Entries, DateTime.Today); int max = Math.Max(1, values.Max(x => Math.Max(x.Breaks, x.Tired)));
        int left = 38, top = 28, bottom = Height - 46, width = Math.Max(1, Width - left - 18), height = Math.Max(1, bottom - top);
        using var font = new Font("Segoe UI", 9); using var brush = new SolidBrush(text); using var line = new Pen(grid);
        for (int i = 0; i <= 4; i++) { int y = bottom - i * height / 4; g.DrawLine(line, left, y, left + width, y); g.DrawString((max * i / 4.0).ToString("0.#"), font, brush, 0, y - 8); }
        float step = width / 7f; float bar = Math.Min(22, step / 3);
        using var green = new SolidBrush(Dark ? Color.FromArgb(155, 163, 255) : Color.FromArgb(79, 86, 170)); using var orange = new SolidBrush(Dark ? Color.FromArgb(255, 186, 128) : Color.FromArgb(145, 77, 28));
        for (int i = 0; i < 7; i++)
        {
            float x = left + i * step + step / 2; float h1 = values[i].Breaks * height / (float)max, h2 = values[i].Tired * height / (float)max;
            g.FillRectangle(green, x - bar - 2, bottom - h1, bar, h1); g.FillRectangle(orange, x + 2, bottom - h2, bar, h2);
            var label = DateTime.Today.AddDays(i - 6).ToString("ddd"); var measured = g.MeasureString(label, font); g.DrawString(label, font, brush, x - measured.Width / 2, bottom + 8);
        }
    }
}
