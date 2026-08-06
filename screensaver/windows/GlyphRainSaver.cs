// Smythe Glyph Rain — native Windows screensaver (.scr).
//
// A GDI+ port of screensaver/index.html: the same 192 framework-generated
// stroke glyphs (see GlyphData.cs, exported by export_glyphs.py), the same
// three-depth-layer digital rain with persistence-fade trails.
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
using System.Runtime.InteropServices;
using System.Windows.Forms;

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
                        "Smythe Glyph Rain\n\n192 procedural cyber glyphs generated as one " +
                        "parallel agent run by the Smythe framework.\n\n" +
                        "github.com/petehottelet/smythe",
                        "Smythe Glyph Rain", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    break;
            }
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
            form.Show();
            Native.SetParent(form.Handle, parent);
            Native.SetWindowLong(form.Handle, Native.GWL_STYLE,
                Native.GetWindowLong(form.Handle, Native.GWL_STYLE) | Native.WS_CHILD);
            Native.MoveWindow(form.Handle, 0, 0,
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
            if (_windowed && ClientSize.Width > 0 && ClientSize.Height > 0 && _buffer != null)
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

            float scale = Math.Max(0.3f, height / 1080f);
            _layers.Clear();
            _layers.Add(new Layer(24f * scale, 0.50f, 0.62f, 0.78f, width, height, _random));
            _layers.Add(new Layer(26f * scale, 0.52f, 0.80f, 0.88f, width, height, _random));
            _layers.Add(new Layer(28f * scale, 0.54f, 1.00f, 1.00f, width, height, _random));
        }

        private void Step()
        {
            DateTime now = DateTime.UtcNow;
            float dt = Math.Min(0.05f, (float)(now - _lastTick).TotalSeconds);
            _lastTick = now;
            // Persistence fade: one shared buffer darkens toward the ground
            // color, so every bright stamp becomes next frame's trail.
            using (var fade = new SolidBrush(Color.FromArgb(10, 0, 5, 2)))
            {
                _graphics.FillRectangle(fade, 0, 0, _buffer.Width, _buffer.Height);
            }
            foreach (Layer layer in _layers)
            {
                layer.Step(_graphics, dt, _random, _buffer.Height);
            }
            Invalidate();
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            e.Graphics.DrawImageUnscaled(_buffer, 0, 0);
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

    /// <summary>One depth plane: pre-rendered sprites plus its column field.</summary>
    internal sealed class Layer : IDisposable
    {
        private readonly Bitmap[] _trail;
        private readonly Bitmap[] _head;
        private readonly List<Column> _columns = new List<Column>();
        private readonly int _step;
        private readonly int _pad;
        private readonly float _speed;
        private readonly float _level;

        internal Layer(float cellF, float spacing, float speed, float level,
                       int width, int height, Random random)
        {
            int cell = Math.Max(6, (int)Math.Round(cellF));
            _speed = speed;
            _level = level;
            _pad = Math.Max(2, cell / 2);
            _trail = new Bitmap[GlyphData.Strokes.Length];
            _head = new Bitmap[GlyphData.Strokes.Length];
            for (int glyph = 0; glyph < GlyphData.Strokes.Length; glyph++)
            {
                _trail[glyph] = Sprites.Render(glyph, cell, _pad, level, head: false);
                _head[glyph] = Sprites.Render(glyph, cell, _pad, level, head: true);
            }
            _step = Math.Max(3, (int)Math.Round(cell * 1.04));
            int lane = Math.Max(3, (int)Math.Round(cell * spacing));
            int columns = (int)Math.Round((Math.Ceiling((double)width / lane) + 1) * 2.0);
            for (int index = 0; index < columns; index++)
            {
                var column = new Column();
                Reset(column, random, height, initial: true);
                column.X = (index * lane) % (width + lane) + random.Next(-3, 4);
                _columns.Add(column);
            }
        }

        private void Reset(Column column, Random random, int height, bool initial)
        {
            column.Glyph = random.Next(GlyphData.Strokes.Length);
            column.Phase = random.Next(GlyphData.Strokes.Length);
            column.Rate = GlyphData.Speeds[column.Glyph] * 5.6f * _speed;
            column.Accumulator = (float)random.NextDouble();
            column.Y = initial
                ? (int)((random.NextDouble() * 2.2 - 1.2) * height)
                : -_step * random.Next(7);
            if (!initial && random.NextDouble() < 0.06)
            {
                column.Burst = 1.6f;
            }
        }

        internal void Step(Graphics graphics, float dt, Random random, int height)
        {
            foreach (Column column in _columns)
            {
                column.Accumulator += dt * column.Rate * (column.Burst > 0 ? 1.9f : 1f);
                if (column.Burst > 0)
                {
                    column.Burst -= dt;
                }
                while (column.Accumulator >= 1f)
                {
                    column.Accumulator -= 1f;
                    Bitmap stamp = _trail[(column.Glyph + column.Phase) % _trail.Length];
                    graphics.DrawImageUnscaled(stamp, column.X - _pad, column.Y - _pad);
                    column.Y += _step;
                    column.Phase = (column.Phase + 7) % _trail.Length;
                    int past = column.Y - GlyphData.Trails[column.Glyph] * _step;
                    if (past > height && random.NextDouble() < 0.6)
                    {
                        Reset(column, random, height, initial: false);
                    }
                }
                if (column.Y > -_step && column.Y < height + _step)
                {
                    Bitmap head = _head[(column.Glyph + column.Phase) % _head.Length];
                    graphics.DrawImageUnscaled(head, column.X - _pad, column.Y - _pad);
                }
            }
        }

        public void Dispose()
        {
            foreach (Bitmap bitmap in _trail)
            {
                bitmap.Dispose();
            }
            foreach (Bitmap bitmap in _head)
            {
                bitmap.Dispose();
            }
        }
    }

    internal sealed class Column
    {
        internal int X;
        internal int Y;
        internal int Glyph;
        internal int Phase;
        internal float Rate;
        internal float Accumulator;
        internal float Burst;
    }

    /// <summary>Pre-renders one glyph's stroke program into a glowing sprite.</summary>
    internal static class Sprites
    {
        internal static Bitmap Render(int glyph, int cell, int pad, float level, bool head)
        {
            int size = cell + pad * 2;
            var bitmap = new Bitmap(size, size, PixelFormat.Format32bppPArgb);
            using (Graphics graphics = Graphics.FromImage(bitmap))
            {
                graphics.SmoothingMode = SmoothingMode.AntiAlias;
                float glyphH = cell;
                float glyphW = glyphH * GlyphData.CanvasW / GlyphData.CanvasH;
                float originX = pad + (cell - glyphW) / 2f;
                float originY = pad;
                List<RectangleF> dots;
                using (GraphicsPath path = BuildPath(glyph, originX, originY, glyphW, glyphH,
                                                     out dots))
                {
                    float unit = glyphW / GlyphData.CanvasW;
                    if (head)
                    {
                        DrawPass(graphics, path, dots, unit, 3.2f,
                                 Color.FromArgb(70, 64, 255, 118));
                        DrawPass(graphics, path, dots, unit, 1.9f,
                                 Color.FromArgb(110, 59, 255, 120));
                        DrawPass(graphics, path, dots, unit, 1.0f,
                                 Color.FromArgb(255, 234, 255, 239));
                    }
                    else
                    {
                        int green = (int)(226 * level);
                        int red = (int)(40 * level);
                        int blue = (int)(96 * level);
                        // Trail sprites carry no glow pass: a fading stamp
                        // dims cleanly rather than leaving a glowing halo
                        // around a dark core.
                        DrawPass(graphics, path, dots, unit, 1.0f,
                                 Color.FromArgb(235, red, green, blue));
                    }
                }
            }
            return bitmap;
        }

        private static void DrawPass(Graphics graphics, GraphicsPath path,
                                     List<RectangleF> dots, float unit,
                                     float widthFactor, Color color)
        {
            using (var pen = new Pen(color))
            {
                pen.LineJoin = LineJoin.Round;
                pen.StartCap = LineCap.Round;
                pen.EndCap = LineCap.Round;
                // GraphicsPath keeps per-subpath widths in metadata we cannot
                // attach, so strokes are drawn via the shared path with an
                // averaged width; the visual difference at rain size is nil.
                pen.Width = Math.Max(0.8f, 7.5f * unit * widthFactor);
                graphics.DrawPath(pen, path);
            }
            using (var brush = new SolidBrush(color))
            {
                foreach (RectangleF dot in dots)
                {
                    RectangleF inflated = dot;
                    inflated.Inflate(dot.Width * (widthFactor - 1f) / 2f,
                                     dot.Height * (widthFactor - 1f) / 2f);
                    graphics.FillEllipse(brush, inflated);
                }
            }
        }

        private static GraphicsPath BuildPath(int glyph, float originX, float originY,
                                              float glyphW, float glyphH,
                                              out List<RectangleF> dots)
        {
            var path = new GraphicsPath();
            dots = new List<RectangleF>();
            float scaleX = glyphW / GlyphData.CanvasW;
            float scaleY = glyphH / GlyphData.CanvasH;
            foreach (float[] stroke in GlyphData.Strokes[glyph])
            {
                int kind = (int)stroke[0];
                if (kind == 2)
                {
                    float radius = Math.Max(0.6f, stroke[3] * scaleX);
                    dots.Add(new RectangleF(
                        originX + stroke[1] * scaleX - radius,
                        originY + stroke[2] * scaleY - radius,
                        radius * 2, radius * 2));
                    continue;
                }
                path.StartFigure();
                float x1 = originX + stroke[1] * scaleX;
                float y1 = originY + stroke[2] * scaleY;
                if (kind == 0)
                {
                    path.AddLine(x1, y1,
                                 originX + stroke[3] * scaleX,
                                 originY + stroke[4] * scaleY);
                }
                else
                {
                    float cx = originX + stroke[3] * scaleX;
                    float cy = originY + stroke[4] * scaleY;
                    float x2 = originX + stroke[5] * scaleX;
                    float y2 = originY + stroke[6] * scaleY;
                    // Quadratic-to-cubic elevation for GDI+ beziers.
                    path.AddBezier(
                        x1, y1,
                        x1 + 2f / 3f * (cx - x1), y1 + 2f / 3f * (cy - y1),
                        x2 + 2f / 3f * (cx - x2), y2 + 2f / 3f * (cy - y2),
                        x2, y2);
                }
            }
            return path;
        }
    }

    internal static class Native
    {
        internal const int GWL_STYLE = -16;
        internal const int WS_CHILD = 0x40000000;

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
