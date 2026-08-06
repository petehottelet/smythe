// Smythe Glyph Rain — native macOS screensaver (.saver).
//
// A CoreGraphics/AppKit port of screensaver/index.html: the same 192
// framework-generated stroke glyphs (Resources/glyphs.json, exported by
// export_glyphs.py), the same three-depth-layer digital rain with
// persistence-fade trails.
//
// Build on a Mac (requires only the Xcode command-line tools):
//     screensaver/macos/build_macos.sh
// then double-click the produced GlyphRain.saver to install.

import AppKit
import ScreenSaver

private struct Catalog {
    let canvasW: CGFloat
    let canvasH: CGFloat
    let strokes: [[[Double]]]
    let speeds: [Double]
    let trails: [Int]

    static func load() -> Catalog? {
        let bundle = Bundle(for: GlyphRainView.self)
        guard let url = bundle.url(forResource: "glyphs", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let canvas = root["canvas"] as? [Double],
              let speeds = root["speeds"] as? [Double],
              let trails = root["trails"] as? [Int],
              let rawStrokes = root["strokes"] as? [[[Any]]]
        else { return nil }
        // Each stroke is ["l"|"q"|"d", numbers...]; normalize to a numeric
        // array with a leading kind code (0 line, 1 quadratic, 2 dot).
        let kinds: [String: Double] = ["l": 0, "q": 1, "d": 2]
        var strokes: [[[Double]]] = []
        strokes.reserveCapacity(rawStrokes.count)
        for glyph in rawStrokes {
            var converted: [[Double]] = []
            for stroke in glyph {
                guard let kind = stroke.first as? String,
                      let code = kinds[kind] else { continue }
                let values = stroke.dropFirst().compactMap { value -> Double? in
                    (value as? NSNumber)?.doubleValue
                }
                converted.append([code] + values)
            }
            strokes.append(converted)
        }
        return Catalog(
            canvasW: CGFloat(canvas[0]),
            canvasH: CGFloat(canvas[1]),
            strokes: strokes,
            speeds: speeds,
            trails: trails
        )
    }
}

private final class Column {
    var x: CGFloat = 0
    var y: CGFloat = 0
    var glyph = 0
    var phase = 0
    var rate: CGFloat = 0
    var accumulator: CGFloat = 0
    var burst: CGFloat = 0
}

private final class Layer {
    let trailSprites: [NSImage]
    let headSprites: [NSImage]
    let pad: CGFloat
    let step: CGFloat
    let speed: CGFloat
    let level: CGFloat
    var columns: [Column] = []

    init(catalog: Catalog, cell: CGFloat, spacing: CGFloat, speed: CGFloat,
         level: CGFloat, width: CGFloat, height: CGFloat) {
        self.speed = speed
        self.level = level
        pad = max(2, cell / 2)
        var trails: [NSImage] = []
        var heads: [NSImage] = []
        for glyph in 0..<catalog.strokes.count {
            trails.append(Layer.sprite(catalog: catalog, glyph: glyph, cell: cell,
                                       pad: pad, level: level, head: false))
            heads.append(Layer.sprite(catalog: catalog, glyph: glyph, cell: cell,
                                      pad: pad, level: level, head: true))
        }
        trailSprites = trails
        headSprites = heads
        step = max(3, cell * 1.04)
        let lane = max(3, cell * spacing)
        let count = Int(((width / lane).rounded(.up) + 1) * 2)
        for index in 0..<count {
            let column = Column()
            column.x = (CGFloat(index) * lane)
                .truncatingRemainder(dividingBy: width + lane)
                + CGFloat.random(in: -3...3)
            column.y = CGFloat.random(in: -1.2...1.0) * height
            reset(column, catalog: catalog, initial: true)
            columns.append(column)
        }
    }

    func reset(_ column: Column, catalog: Catalog, initial: Bool) {
        column.glyph = Int.random(in: 0..<catalog.strokes.count)
        column.phase = Int.random(in: 0..<catalog.strokes.count)
        column.rate = CGFloat(catalog.speeds[column.glyph]) * 5.6 * speed
        column.accumulator = CGFloat.random(in: 0...1)
        if !initial {
            column.y = -step * CGFloat(Int.random(in: 0..<7))
            if Double.random(in: 0...1) < 0.06 { column.burst = 1.6 }
        }
    }

    private static func sprite(catalog: Catalog, glyph: Int, cell: CGFloat,
                               pad: CGFloat, level: CGFloat, head: Bool) -> NSImage {
        let size = cell + pad * 2
        let image = NSImage(size: NSSize(width: size, height: size))
        image.lockFocus()
        if let context = NSGraphicsContext.current?.cgContext {
            // Flip so glyph y grows downward like every other port.
            context.translateBy(x: 0, y: size)
            context.scaleBy(x: 1, y: -1)
            let glyphH = cell
            let glyphW = glyphH * catalog.canvasW / catalog.canvasH
            let originX = pad + (cell - glyphW) / 2
            let originY = pad
            let unit = glyphW / catalog.canvasW
            if head {
                context.setShadow(offset: .zero, blur: cell * 0.42,
                                  color: CGColor(red: 0.27, green: 1.0, blue: 0.51,
                                                 alpha: 0.95))
                draw(catalog: catalog, glyph: glyph, in: context,
                     originX: originX, originY: originY, unit: unit,
                     color: CGColor(red: 0.92, green: 1.0, blue: 0.94, alpha: 1))
            } else {
                context.setShadow(offset: .zero, blur: 0, color: nil)
                draw(catalog: catalog, glyph: glyph, in: context,
                     originX: originX, originY: originY, unit: unit,
                     color: CGColor(red: 0.157 * level, green: 0.886 * level,
                                    blue: 0.376 * level, alpha: 0.92))
            }
        }
        image.unlockFocus()
        return image
    }

    private static func draw(catalog: Catalog, glyph: Int, in context: CGContext,
                             originX: CGFloat, originY: CGFloat, unit: CGFloat,
                             color: CGColor) {
        context.setLineCap(.round)
        context.setLineJoin(.round)
        context.setStrokeColor(color)
        context.setFillColor(color)
        for stroke in catalog.strokes[glyph] {
            let kind = Int(stroke[0])
            if kind == 2 {
                let radius = max(0.8, CGFloat(stroke[3]) * unit)
                let rect = CGRect(
                    x: originX + CGFloat(stroke[1]) * unit - radius,
                    y: originY + CGFloat(stroke[2]) * unit - radius,
                    width: radius * 2, height: radius * 2)
                context.fillEllipse(in: rect)
                continue
            }
            context.setLineWidth(max(1.2, CGFloat(stroke.last ?? 8) * unit))
            context.beginPath()
            context.move(to: CGPoint(x: originX + CGFloat(stroke[1]) * unit,
                                     y: originY + CGFloat(stroke[2]) * unit))
            if kind == 0 {
                context.addLine(to: CGPoint(x: originX + CGFloat(stroke[3]) * unit,
                                            y: originY + CGFloat(stroke[4]) * unit))
            } else {
                context.addQuadCurve(
                    to: CGPoint(x: originX + CGFloat(stroke[5]) * unit,
                                y: originY + CGFloat(stroke[6]) * unit),
                    control: CGPoint(x: originX + CGFloat(stroke[3]) * unit,
                                     y: originY + CGFloat(stroke[4]) * unit))
            }
            context.strokePath()
        }
    }
}

@objc(GlyphRainView)
public final class GlyphRainView: ScreenSaverView {
    private var catalog: Catalog?
    private var layers: [Layer] = []
    private var buffer: NSImage?
    private var lastTick = Date()

    public override init?(frame: NSRect, isPreview: Bool) {
        super.init(frame: frame, isPreview: isPreview)
        animationTimeInterval = 1.0 / 40.0
    }

    public required init?(coder: NSCoder) {
        super.init(coder: coder)
        animationTimeInterval = 1.0 / 40.0
    }

    public override func startAnimation() {
        super.startAnimation()
        buildScene()
    }

    private func buildScene() {
        guard let catalog = catalog ?? Catalog.load() else { return }
        self.catalog = catalog
        let width = bounds.width
        let height = bounds.height
        let scale = max(0.3, height / 1080)
        layers = [
            Layer(catalog: catalog, cell: 24 * scale, spacing: 0.50, speed: 0.62,
                  level: 0.78, width: width, height: height),
            Layer(catalog: catalog, cell: 26 * scale, spacing: 0.52, speed: 0.80,
                  level: 0.88, width: width, height: height),
            Layer(catalog: catalog, cell: 28 * scale, spacing: 0.54, speed: 1.00,
                  level: 1.00, width: width, height: height),
        ]
        let image = NSImage(size: bounds.size)
        image.lockFocus()
        NSColor(calibratedRed: 0, green: 0.02, blue: 0.008, alpha: 1).setFill()
        bounds.fill()
        image.unlockFocus()
        buffer = image
    }

    public override func animateOneFrame() {
        guard let catalog = catalog, let buffer = buffer else {
            buildScene()
            needsDisplay = true
            return
        }
        let now = Date()
        let dt = CGFloat(min(0.05, now.timeIntervalSince(lastTick)))
        lastTick = now
        let height = bounds.height
        buffer.lockFocus()
        // Persistence fade toward the ground color.
        NSColor(calibratedRed: 0, green: 0.02, blue: 0.008, alpha: 0.062).setFill()
        bounds.fill(using: .sourceOver)
        for layer in layers {
            for column in layer.columns {
                column.accumulator += dt * column.rate * (column.burst > 0 ? 1.9 : 1)
                if column.burst > 0 { column.burst -= dt }
                while column.accumulator >= 1 {
                    column.accumulator -= 1
                    let stamp = layer.trailSprites[
                        (column.glyph + column.phase) % layer.trailSprites.count]
                    // The buffer's coordinate space is bottom-up; convert the
                    // top-down simulation y.
                    stamp.draw(at: NSPoint(x: column.x - layer.pad,
                                           y: height - column.y - layer.pad),
                               from: .zero, operation: .sourceOver, fraction: 1)
                    column.y += layer.step
                    column.phase = (column.phase + 7) % layer.trailSprites.count
                    let past = column.y
                        - CGFloat(catalog.trails[column.glyph]) * layer.step
                    if past > height, Double.random(in: 0...1) < 0.6 {
                        layer.reset(column, catalog: catalog, initial: false)
                    }
                }
                if column.y > -layer.step, column.y < height + layer.step {
                    let head = layer.headSprites[
                        (column.glyph + column.phase) % layer.headSprites.count]
                    head.draw(at: NSPoint(x: column.x - layer.pad,
                                          y: height - column.y - layer.pad),
                              from: .zero, operation: .sourceOver,
                              fraction: 0.92 * layer.level)
                }
            }
        }
        buffer.unlockFocus()
        needsDisplay = true
    }

    public override func draw(_ rect: NSRect) {
        NSColor.black.setFill()
        rect.fill()
        buffer?.draw(in: bounds, from: .zero, operation: .sourceOver, fraction: 1)
    }

    public override var hasConfigureSheet: Bool { false }
    public override var configureSheet: NSWindow? { nil }
}
