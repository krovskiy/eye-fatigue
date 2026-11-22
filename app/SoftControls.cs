using System.Drawing.Drawing2D;

namespace EyeFatigue;

internal sealed class SoftButton : Button
{
    bool hover;
    public SoftButton() { SetStyle(ControlStyles.UserPaint | ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer, true); FlatStyle = FlatStyle.Flat; FlatAppearance.BorderSize = 0; }
    protected override void OnMouseEnter(EventArgs e) { hover = true; Invalidate(); base.OnMouseEnter(e); }
    protected override void OnMouseLeave(EventArgs e) { hover = false; Invalidate(); base.OnMouseLeave(e); }
    protected override void OnPaint(PaintEventArgs e)
    {
        e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
        var rect = new Rectangle(1, 1, Width - 3, Height - 3);
        using var path = Rounded(rect, 9);
        using var fill = new SolidBrush(hover ? ControlPaint.Light(BackColor, .12f) : BackColor);
        e.Graphics.FillPath(fill, path);
        TextRenderer.DrawText(e.Graphics, Text, Font, rect, ForeColor, TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter | TextFormatFlags.EndEllipsis | TextFormatFlags.NoPrefix);
        if (Focused && ShowFocusCues) { using var pen = new Pen(ForeColor, 2); e.Graphics.DrawPath(pen, path); }
    }
    internal static GraphicsPath Rounded(Rectangle r, int radius)
    {
        var p = new GraphicsPath(); int d = Math.Min(radius * 2, Math.Min(r.Width, r.Height));
        p.AddArc(r.Left, r.Top, d, d, 180, 90); p.AddArc(r.Right - d, r.Top, d, d, 270, 90);
        p.AddArc(r.Right - d, r.Bottom - d, d, d, 0, 90); p.AddArc(r.Left, r.Bottom - d, d, d, 90, 90); p.CloseFigure(); return p;
    }
}

internal sealed class SoftProgress : Control
{
    int value;
    [System.ComponentModel.DesignerSerializationVisibility(System.ComponentModel.DesignerSerializationVisibility.Hidden)]
    public int Value { get => value; set { this.value = Math.Clamp(value, 0, 1000); Invalidate(); } }
    public Color Track = Color.FromArgb(60, 61, 69);
    public Color Fill = Color.FromArgb(155, 163, 255);
    public SoftProgress() { DoubleBuffered = true; Height = 10; AccessibleName = "Time until your next break"; AccessibleRole = AccessibleRole.ProgressBar; }
    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e); if (Width < 2 || Height < 2) return; e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
        using var track = new SolidBrush(Track); using var fill = new SolidBrush(Fill); using var outline = SoftButton.Rounded(new Rectangle(0, 0, Width - 1, Height - 1), Height / 2);
        e.Graphics.FillPath(track, outline); int w = (Width - 1) * value / 1000;
        if (w > 0) { e.Graphics.SetClip(outline); e.Graphics.FillRectangle(fill, 0, 0, w, Height); e.Graphics.ResetClip(); }
    }
}
