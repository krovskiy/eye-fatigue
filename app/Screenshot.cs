namespace EyeFatigue;

internal static class Screenshot
{
    public static void Capture(string directory)
    {
        Directory.CreateDirectory(directory);
        using var form = new MainForm(demo: true);
        form.Show(); Application.DoEvents();
        foreach (var (index, name) in new[] { (0, "dashboard.png"), (1, "activity.png"), (2, "settings.png") })
        {
            form.SelectPage(index); Application.DoEvents();
            using var image = new Bitmap(form.Width, form.Height);
            form.DrawToBitmap(image, new Rectangle(Point.Empty, form.Size));
            image.Save(Path.Combine(directory, name), System.Drawing.Imaging.ImageFormat.Png);
        }
        form.Close();
    }
}
