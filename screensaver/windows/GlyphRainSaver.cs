// Smythe Glyph Rain — native Windows screensaver (.scr).
//
// Exact filled SVG catalogs: 57 reference slots (including one blank) and
// 192 original glyphs. Cached compound-path sprites use a 90/10 cell mix.
//
// Build (no SDK needed beyond Windows' bundled .NET Framework compiler):
//     screensaver\windows\build_windows.cmd
//
// Screensaver argument convention:
//     /s          run fullscreen on every monitor
//     /p <hwnd>   render inside the settings-dialog preview window
//     /c          show the about box (no settings)
//     /w          run in a resizable window (debugging)

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Windows.Forms;

[assembly: AssemblyTitle("Smythe Glyph Rain")]
[assembly: AssemblyDescription("Native filled SVG code rain: 56 reference shapes and 192 original glyphs")]
[assembly: AssemblyVersion("1.1.0.0")]
[assembly: AssemblyFileVersion("1.1.0.0")]

namespace SmytheGlyphRain
{
    internal static class Program
    {
        [STAThread]
        private static void Main(string[] args)
        {
            Application.EnableVisualStyles();
            string mode = args.Length > 0 ? args[0].Trim().ToLowerInvariant() : "/c";
            long handle;
            if (mode.StartsWith("/p") && args.Length > 1 && long.TryParse(args[1], out handle))
            {
                RunPreview(new IntPtr(handle));
                return;
            }
            switch (mode)
            {
                case "/s":
                    RunFullscreen();
                    break;
                case "/w":
                    Application.Run(new SaverForm(new Rectangle(80, 80, 1280, 720), windowed: true));
                    break;
                default:
                    MessageBox.Show(
                        "Smythe Glyph Rain\n\n56 reference SVG shapes + 192 original SVG glyphs.\n" +
                        "90% reference sequence / 10% originals.\n\n" +
                        "github.com/petehottelet/smythe\n\n" + ReferenceLicense(),
                        "Smythe Glyph Rain", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    break;
            }
        }

        private static string ReferenceLicense()
        {
            using (Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("SmytheGlyphRain.ReferenceLicense"))
            using (var reader = new StreamReader(stream)) { return reader.ReadToEnd(); }
        }

        private static void RunFullscreen()
        {
            var forms = new List<SaverForm>();
            foreach (Screen screen in Screen.AllScreens)
            {
                forms.Add(new SaverForm(screen.Bounds, windowed: false));
            }
            foreach (SaverForm form in forms)
            {
                form.Show();
            }
            Application.Run();
        }

        private static void RunPreview(IntPtr parent)
        {
            Native.RECT rect;
            if (!Native.GetClientRect(parent, out rect))
            {
                return;
            }
            var form = new SaverForm(
                new Rectangle(0, 0, rect.Right - rect.Left, rect.Bottom - rect.Top),
                windowed: false, preview: true);
            form.TopLevel = false;
            // Creating the handle keeps the form hidden. Attach it before the
            // message loop shows it, so Settings preview cannot flash a desktop
            // window and inherits its parent's visibility from its first frame.
            IntPtr child = form.Handle;
            Native.SetWindowLong(form.Handle, Native.GWL_STYLE,
                (Native.GetWindowLong(child, Native.GWL_STYLE) & ~Native.WS_POPUP) | Native.WS_CHILD);
            Native.SetParent(child, parent);
            Native.MoveWindow(child, 0, 0,
                rect.Right - rect.Left, rect.Bottom - rect.Top, true);
            Application.Run(form);
        }
    }

    internal sealed class SaverForm : Form
    {
        private readonly bool _windowed;
        private readonly bool _preview;
        private readonly Random _random = new Random();
        private readonly List<Layer> _layers = new List<Layer>();
        private Bitmap _buffer;
        private Graphics _graphics;
        private Timer _timer;
        private Point _lastMouse = Point.Empty;
        private DateTime _lastTick = DateTime.UtcNow;

        internal SaverForm(Rectangle bounds, bool windowed, bool preview = false)
        {
            _windowed = windowed;
            _preview = preview;
            SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint |
                     ControlStyles.OptimizedDoubleBuffer, true);
            BackColor = Color.FromArgb(0, 5, 2);
            if (windowed)
            {
                Text = "Smythe Glyph Rain (windowed debug)";
                StartPosition = FormStartPosition.Manual;
                Bounds = bounds;
            }
            else
            {
                FormBorderStyle = FormBorderStyle.None;
                StartPosition = FormStartPosition.Manual;
                Bounds = bounds;
                if (!preview)
                {
                    TopMost = true;
                    Cursor.Hide();
                }
            }
        }

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);
            BuildScene();
            _timer = new Timer { Interval = 25 };
            _timer.Tick += (_, __) => Step();
            _timer.Start();
        }

        protected override void OnResize(EventArgs e)
        {
            base.OnResize(e);
            if (ClientSize.Width > 0 && ClientSize.Height > 0 && _buffer != null)
            {
                BuildScene();
            }
        }

        private void BuildScene()
        {
            if (_graphics != null) { _graphics.Dispose(); }
            if (_buffer != null) { _buffer.Dispose(); }
            int width = Math.Max(1, ClientSize.Width);
            int height = Math.Max(1, ClientSize.Height);
            _buffer = new Bitmap(width, height, PixelFormat.Format32bppPArgb);
            _graphics = Graphics.FromImage(_buffer);
            _graphics.Clear(Color.FromArgb(0, 5, 2));
            _graphics.InterpolationMode = InterpolationMode.Bilinear;

            float scale = Math.Max(0.5f, Math.Min(1.5f, height / 720f));
            foreach (Layer layer in _layers) { layer.Dispose(); }
            _layers.Clear();
            _layers.Add(new Layer(11f * scale, 1.20f, 0.62f, 0.48f, width, height, _random));
            _layers.Add(new Layer(20f * scale, 1.35f, 0.80f, 0.75f, width, height, _random));
            _layers.Add(new Layer(36f * scale, 2.50f, 1.00f, 1.00f, width, height, _random));
            DrawFrame(0);
            _lastTick = DateTime.UtcNow;
        }

        private void Step()
        {
            DateTime now = DateTime.UtcNow;
            float dt = Math.Min(0.05f, (float)(now - _lastTick).TotalSeconds);
            _lastTick = now;
            DrawFrame(dt);
            Invalidate();
        }

        private void DrawFrame(float dt)
        {
            _graphics.Clear(Color.Black);
            foreach (Layer layer in _layers)
            {
                layer.Step(_graphics, dt, _random, _buffer.Height);
            }
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            if (_buffer != null) { e.Graphics.DrawImageUnscaled(_buffer, 0, 0); }
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            if (_windowed || _preview)
            {
                return;
            }
            if (_lastMouse == Point.Empty)
            {
                _lastMouse = e.Location;
                return;
            }
            if (Math.Abs(e.X - _lastMouse.X) + Math.Abs(e.Y - _lastMouse.Y) > 4)
            {
                Application.Exit();
            }
        }

        protected override void OnMouseDown(MouseEventArgs e)
        {
            if (!_windowed && !_preview)
            {
                Application.Exit();
            }
        }

        protected override void OnKeyDown(KeyEventArgs e)
        {
            if (!_preview && (!_windowed || e.KeyCode == Keys.Escape))
            {
                Application.Exit();
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (_timer != null) { _timer.Dispose(); }
                if (_graphics != null) { _graphics.Dispose(); }
                if (_buffer != null) { _buffer.Dispose(); }
                foreach (Layer layer in _layers)
                {
                    layer.Dispose();
                }
            }
            base.Dispose(disposing);
        }
    }

    /// <summary>One depth plane of cached exact SVG fills.</summary>
    internal sealed class Layer : IDisposable
    {
        private readonly Bitmap[] _trail;
        private readonly Bitmap[] _head;
        private readonly List<Column> _columns = new List<Column>();
        private readonly int _step;
        private readonly int _pad;
        private readonly float _speed;
        private readonly ImageAttributes[] _opacity = new ImageAttributes[16];
        internal long DrawnReference;
        internal long DrawnOriginal;
        internal long DrawnBlank;

        internal Layer(float cellF, float spacing, float speed, float level,
                       int width, int height, Random random)
        {
            int cell = Math.Max(6, (int)Math.Round(cellF));
            _speed = speed;
            _pad = Math.Max(3, cell / 2);
            _trail = new Bitmap[GlyphData.Count];
            _head = new Bitmap[GlyphData.Count];
            for (int alpha = 0; alpha < _opacity.Length; alpha++)
            {
                _opacity[alpha] = new ImageAttributes();
                var matrix = new ColorMatrix();
                matrix.Matrix33 = alpha / 15f;
                _opacity[alpha].SetColorMatrix(matrix);
            }
            for (int glyph = 0; glyph < GlyphData.Count; glyph++)
            {
                _trail[glyph] = Sprites.Render(glyph, cell, _pad, level, false);
                _head[glyph] = Sprites.Render(glyph, cell, _pad, level, true);
            }
            _step = Math.Max(3, (int)Math.Round(cell * 1.04));
            int lane = Math.Max(3, (int)Math.Round(cell * spacing));
            int columns = (int)Math.Ceiling((double)width / lane) + 1;
            for (int index = 0; index < columns; index++)
            {
                var column = new Column();
                Reset(column, random, height, true);
                column.X = index * lane + random.Next(-3, 4);
                _columns.Add(column);
            }
        }

        private void Reset(Column column, Random random, int height, bool initial)
        {
            column.Seed = random.Next();
            column.Glyph = CatalogSelector.Select(column.Seed, 0, 0);
            column.Phase = random.Next(1000);
            column.Rate = (float)GlyphData.Speeds[column.Glyph] * 5.6f * _speed;
            column.Accumulator = (float)random.NextDouble();
            column.Y = initial
                ? (int)(random.NextDouble() * (height + GlyphData.Trails[column.Glyph] * _step))
                : -_step * random.Next(7);
            column.Burst = !initial && random.NextDouble() < 0.06 ? 1.6f : 0;
        }

        internal void Step(Graphics graphics, float dt, Random random, int height)
        {
            foreach (Column column in _columns)
            {
                column.Accumulator += dt * column.Rate * (column.Burst > 0 ? 1.9f : 1f);
                if (column.Burst > 0) { column.Burst -= dt; }
                while (column.Accumulator >= 1f)
                {
                    column.Accumulator -= 1f;
                    column.Y += _step;
                    column.Phase++;
                    if (column.Y - GlyphData.Trails[column.Glyph] * _step * 1.15f > height
                        && random.NextDouble() < 0.6)
                    {
                        Reset(column, random, height, false);
                    }
                }
                int length = (int)Math.Round(GlyphData.Trails[column.Glyph] * 1.15f);
                for (int tail = length; tail >= 0; tail--)
                {
                    int y = column.Y - tail * _step;
                    if (y < -_step || y > height + _step) { continue; }
                    // A catalog coin flip per cell keeps 192 originals from
                    // overwhelming the reference's 57-slot sequence.
                    int glyph = CatalogSelector.Select(column.Seed, y / _step, column.Phase / 6);
                    if (glyph == GlyphData.BlankIndex) { DrawnBlank++; continue; }
                    if (glyph < GlyphData.OriginalOffset) { DrawnReference++; }
                    else { DrawnOriginal++; }
                    float near = 1f - tail / (float)length;
                    float shimmer = 0.75f + (glyph % 5) * 0.0625f;
                    int alpha = tail == 0 ? 15
                        : (int)Math.Round((0.25 + 0.75 * Math.Sqrt(near)) * shimmer * 15);
                    Bitmap sprite = tail == 0 ? _head[glyph] : _trail[glyph];
                    graphics.DrawImage(sprite,
                        new Rectangle(column.X - _pad, y - _pad, sprite.Width, sprite.Height),
                        0, 0, sprite.Width, sprite.Height, GraphicsUnit.Pixel, _opacity[alpha]);
                }
            }
        }

        public void Dispose()
        {
            foreach (ImageAttributes opacity in _opacity) { opacity.Dispose(); }
            foreach (Bitmap bitmap in _trail) { bitmap.Dispose(); }
            foreach (Bitmap bitmap in _head) { bitmap.Dispose(); }
        }
    }

    internal sealed class Column
    {
        internal int X, Y, Glyph, Phase, Seed;
        internal float Rate, Accumulator, Burst;
    }

    internal static class CatalogSelector
    {
        private static uint Mix(uint value)
        {
            unchecked {
                value ^= value >> 16; value *= 0x7feb352du;
                value ^= value >> 15; value *= 0x846ca68bu;
                return value ^ (value >> 16);
            }
        }

        internal static int Select(int seed, int row, int epoch)
        {
            unchecked {
                uint hash = Mix((uint)seed ^ ((uint)row * 0x9e3779b9u) ^ ((uint)epoch * 0x85ebca6bu));
                uint index = Mix(hash ^ 0xa511e9b3u);
                return hash % 10000 < (uint)(GlyphData.OriginalShare * 10000)
                    ? GlyphData.OriginalOffset + (int)(index % (uint)GlyphData.OriginalCount)
                    : (int)(index % (uint)GlyphData.ReferenceCount);
            }
        }
    }

    internal static class Sprites
    {
        // GDI+ uses coarse edge coverage at small sizes. Average a 4x cached
        // raster so the same SVG area covers the same screen pixels as Cairo/CG.
        private const int RasterScale = 4;
        internal static GraphicsPath BuildPath(int glyph)
        {
            return BuildPath(GlyphData.Commands[glyph]);
        }

        internal static GraphicsPath BuildPath(double[][] commands)
        {
            var path = new GraphicsPath(FillMode.Winding);
            float x = 0, y = 0;
            bool open = false;
            try {
                foreach (double[] command in commands)
                {
                    if (command.Length == 0) { throw new InvalidOperationException("Empty SVG command"); }
                    int kind = (int)command[0];
                    int expected = kind == 2 ? 7 : kind == 3 ? 1 : 3;
                    if (kind < 0 || kind > 3 || command[0] != kind || command.Length != expected)
                    { throw new InvalidOperationException("Invalid SVG command"); }
                    foreach (double value in command)
                        if (Double.IsNaN(value) || Double.IsInfinity(value))
                            throw new InvalidOperationException("Non-finite SVG coordinate");
                    if (kind == 0)
                    {
                        if (open) { throw new InvalidOperationException("Unclosed SVG contour"); }
                        path.StartFigure(); x = (float)command[1]; y = (float)command[2]; open = true;
                    }
                    else
                    {
                        if (!open) { throw new InvalidOperationException("SVG command before move"); }
                        if (kind == 1)
                        {
                            path.AddLine(x, y, (float)command[1], (float)command[2]);
                            x = (float)command[1]; y = (float)command[2];
                        }
                        else if (kind == 2)
                        {
                            path.AddBezier(x, y, (float)command[1], (float)command[2],
                                (float)command[3], (float)command[4], (float)command[5], (float)command[6]);
                            x = (float)command[5]; y = (float)command[6];
                        }
                        else { path.CloseFigure(); open = false; }
                    }
                }
                if (open) { throw new InvalidOperationException("Unclosed SVG contour"); }
                return path;
            }
            catch { path.Dispose(); throw; }
        }

        internal static Bitmap RenderMask(int glyph, int cell)
        {
            var bitmap = new Bitmap(cell * RasterScale, cell * RasterScale, PixelFormat.Format32bppPArgb);
            using (Graphics graphics = Graphics.FromImage(bitmap))
            using (GraphicsPath path = BuildPath(glyph))
            using (var transform = new Matrix(cell * RasterScale / GlyphData.CanvasW, 0, 0, cell * RasterScale / GlyphData.CanvasH, 0, 0))
            {
                graphics.SmoothingMode = SmoothingMode.AntiAlias;
                graphics.PixelOffsetMode = PixelOffsetMode.Half;
                path.Transform(transform);
                graphics.FillPath(Brushes.White, path);
            }
            return Downsample(bitmap);
        }

        internal static Bitmap Render(int glyph, int cell, int pad, float level, bool head)
        {
            var bitmap = new Bitmap((cell + pad * 2) * RasterScale, (cell + pad * 2) * RasterScale, PixelFormat.Format32bppPArgb);
            using (Graphics graphics = Graphics.FromImage(bitmap))
            using (GraphicsPath path = BuildPath(glyph))
            using (var transform = new Matrix(cell * RasterScale / GlyphData.CanvasW, 0, 0, cell * RasterScale / GlyphData.CanvasH, pad * RasterScale, pad * RasterScale))
            {
                graphics.SmoothingMode = SmoothingMode.AntiAlias;
                graphics.PixelOffsetMode = PixelOffsetMode.Half;
                path.Transform(transform);
                Color body = BodyColor(level);
                Color color = head ? Color.FromArgb(255, (int)(162 * level), (int)(255 * level), (int)(216 * level)) : body;
                // Low-opacity outlines form a halo; the opaque core is the exact
                // compound SVG fill, never a stroke-width approximation.
                using (var wide = new Pen(Color.FromArgb((int)(14 * level), 117, 240, 152), Math.Max(1f, cell * 0.20f) * RasterScale))
                using (var soft = new Pen(Color.FromArgb((int)(head ? 35 * level : 20 * level), 117, 240, 152), Math.Max(1f, cell * 0.08f) * RasterScale))
                using (var brush = new SolidBrush(color))
                {
                    wide.LineJoin = soft.LineJoin = LineJoin.Round;
                    graphics.DrawPath(wide, path); graphics.DrawPath(soft, path);
                    graphics.FillPath(brush, path);
                }
            }
            return Downsample(bitmap);
        }

        private static Bitmap Downsample(Bitmap source)
        {
            var result = new Bitmap(source.Width / RasterScale, source.Height / RasterScale, PixelFormat.Format32bppPArgb);
            BitmapData input = null, output = null;
            bool completed = false;
            try {
                input = source.LockBits(new Rectangle(0, 0, source.Width, source.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppPArgb);
                output = result.LockBits(new Rectangle(0, 0, result.Width, result.Height), ImageLockMode.WriteOnly, PixelFormat.Format32bppPArgb);
                byte[] pixels = new byte[input.Stride * source.Height], averaged = new byte[output.Stride * result.Height];
                Marshal.Copy(input.Scan0, pixels, 0, pixels.Length);
                const int samples = RasterScale * RasterScale;
                for (int y = 0; y < result.Height; y++) for (int x = 0; x < result.Width; x++)
                    for (int channel = 0; channel < 4; channel++)
                    {
                        int sum = 0;
                        for (int dy = 0; dy < RasterScale; dy++) for (int dx = 0; dx < RasterScale; dx++)
                            sum += pixels[(y * RasterScale + dy) * input.Stride + (x * RasterScale + dx) * 4 + channel];
                        averaged[y * output.Stride + x * 4 + channel] = (byte)((sum + samples / 2) / samples);
                    }
                Marshal.Copy(averaged, 0, output.Scan0, averaged.Length);
                completed = true;
                return result;
            }
            finally {
                if (input != null) source.UnlockBits(input);
                if (output != null) result.UnlockBits(output);
                source.Dispose();
                if (!completed) result.Dispose();
            }
        }

        private static Color BodyColor(float level)
        {
            double lightness = 0.7 * level;
            double chroma = (1 - Math.Abs(2 * lightness - 1)) * 0.8;
            double secondary = chroma * (1 - Math.Abs((137.0 / 60) % 2 - 1));
            double m = lightness - chroma / 2;
            return Color.FromArgb(255, (int)Math.Round(m * 255),
                (int)Math.Round((chroma + m) * 255), (int)Math.Round((secondary + m) * 255));
        }
    }

    internal static class CatalogDiagnostics
    {
        internal static Dictionary<string, object> Verify()
        {
            if (GlyphData.Count != 249 || GlyphData.ReferenceCount != 57 || GlyphData.ReferenceVisibleCount != 56
                || GlyphData.OriginalCount != 192 || GlyphData.OriginalOffset != 57 || GlyphData.BlankIndex != 4
                || GlyphData.Commands.Length != 249 || GlyphData.GlyphIds.Length != 249 || GlyphData.FillRule != "nonzero")
                throw new InvalidOperationException("Incorrect filled SVG catalog");
            int curves = 0, referenceCounters = 0, originalCounters = 0, visible = 0;
            var identities = new HashSet<string>();
            for (int glyph = 0; glyph < GlyphData.Count; glyph++)
            {
                if (!identities.Add(GlyphData.GlyphIds[glyph])) throw new InvalidOperationException("Duplicate glyph identity");
                foreach (double[] command in GlyphData.Commands[glyph]) if (command[0] == 2) curves++;
                using (Bitmap mask = Sprites.RenderMask(glyph, 64))
                {
                    int ink, holes;
                    InspectMask(mask, out ink, out holes);
                    if (glyph == GlyphData.BlankIndex)
                    { if (ink != 0 || GlyphData.Commands[glyph].Length != 0) throw new InvalidOperationException("Reference blank was lost"); }
                    else
                    {
                        if (ink == 0) throw new InvalidOperationException("Empty filled glyph " + GlyphData.GlyphIds[glyph]);
                        visible++;
                        if (holes > 0) { if (glyph < GlyphData.OriginalOffset) referenceCounters++; else originalCounters++; }
                    }
                }
            }
            int originalSamples = 0, blankSamples = 0;
            for (int i = 0; i < 100000; i++)
            {
                int glyph = CatalogSelector.Select(7319, i, 0);
                if (glyph >= GlyphData.OriginalOffset) originalSamples++;
                if (glyph == GlyphData.BlankIndex) blankSamples++;
            }
            if (visible != 248 || curves == 0 || referenceCounters == 0 || originalCounters == 0
                || originalSamples < 9500 || originalSamples > 10500 || blankSamples == 0)
                throw new InvalidOperationException("Filled SVG geometry or weighted selection failed");
            return new Dictionary<string, object> {
                {"catalog_sha256", GlyphData.CatalogSha256}, {"reference_sha256", GlyphData.ReferenceSha256},
                {"original_sha256", GlyphData.OriginalSha256}, {"count", GlyphData.Count}, {"visible_count", visible},
                {"reference_sequence_count", GlyphData.ReferenceCount}, {"original_count", GlyphData.OriginalCount},
                {"blank_index", GlyphData.BlankIndex}, {"first_original_id", GlyphData.GlyphIds[GlyphData.OriginalOffset]},
                {"fill_rule", "nonzero"}, {"cubic_commands", curves},
                {"reference_glyphs_with_counters_at_64px", referenceCounters}, {"original_glyphs_with_counters_at_64px", originalCounters},
                {"selection_samples", 100000}, {"original_selections", originalSamples}, {"blank_selections", blankSamples}
            };
        }

        internal static Bitmap Atlas()
        {
            const int cell = 64, columns = 16;
            var image = new Bitmap(columns * cell, ((GlyphData.Count + columns - 1) / columns) * cell);
            using (Graphics graphics = Graphics.FromImage(image))
            {
                graphics.Clear(Color.Black);
                for (int glyph = 0; glyph < GlyphData.Count; glyph++)
                    using (Bitmap mask = Sprites.RenderMask(glyph, cell))
                        graphics.DrawImageUnscaled(mask, glyph % columns * cell, glyph / columns * cell);
            }
            return image;
        }

        private static void InspectMask(Bitmap bitmap, out int ink, out int holes)
        {
            int width = bitmap.Width, height = bitmap.Height;
            bool[] filled = new bool[width * height], visited = new bool[width * height];
            ink = 0; holes = 0;
            for (int y = 0; y < height; y++) for (int x = 0; x < width; x++)
                if (bitmap.GetPixel(x, y).A >= 128) { filled[y * width + x] = true; ink++; }
            var queue = new Queue<int>();
            for (int start = 0; start < filled.Length; start++)
            {
                if (filled[start] || visited[start]) continue;
                bool edge = false; int area = 0;
                queue.Enqueue(start); visited[start] = true;
                while (queue.Count > 0)
                {
                    int p = queue.Dequeue(), x = p % width, y = p / width; area++;
                    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) edge = true;
                    int[] neighbors = { x > 0 ? p - 1 : -1, x < width - 1 ? p + 1 : -1,
                        y > 0 ? p - width : -1, y < height - 1 ? p + width : -1 };
                    foreach (int n in neighbors) if (n >= 0 && !filled[n] && !visited[n])
                    { visited[n] = true; queue.Enqueue(n); }
                }
                if (!edge && area >= 4) holes++;
            }
        }
    }

    internal static class Native
    {
        internal const int GWL_STYLE = -16;
        internal const int WS_CHILD = 0x40000000;
        internal const int WS_POPUP = unchecked((int)0x80000000);

        [StructLayout(LayoutKind.Sequential)]
        internal struct RECT
        {
            public int Left;
            public int Top;
            public int Right;
            public int Bottom;
        }

        [DllImport("user32.dll")]
        internal static extern bool GetClientRect(IntPtr hWnd, out RECT rect);

        [DllImport("user32.dll")]
        internal static extern IntPtr SetParent(IntPtr child, IntPtr parent);

        [DllImport("user32.dll")]
        internal static extern int GetWindowLong(IntPtr hWnd, int index);

        [DllImport("user32.dll")]
        internal static extern int SetWindowLong(IntPtr hWnd, int index, int value);

        [DllImport("user32.dll")]
        internal static extern bool MoveWindow(IntPtr hWnd, int x, int y,
                                               int width, int height, bool repaint);
    }
}
