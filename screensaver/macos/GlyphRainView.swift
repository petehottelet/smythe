// Smythe Glyph Rain: exact compound SVG fills in a native ScreenSaverView.
// Reference sequence: 56 shapes plus one blank; original catalog: 192 shapes.
import AppKit
import ScreenSaver

private struct Catalog {
    let commands: [[[Double]]]
    let paths: [CGPath]
    let speeds: [Double]
    let trails: [Int]
    let ids: [String]
    let catalogHash: String
    let referenceHash: String
    let originalHash: String
    let originalShare: Double
    static let referenceCount = 57
    static let originalOffset = 57
    static let originalCount = 192
    static let blankIndex = 4

    static func load() -> Catalog? {
        let bundle = Bundle(for: GlyphRainView.self)
        guard let url = bundle.url(forResource: "glyphs", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let commands = root["commands"] as? [[[Double]]], commands.count == 249,
              let canvas = root["canvas"] as? [Double], canvas == [100, 100],
              let speeds = root["speeds"] as? [Double], speeds.count == 249,
              let trails = root["trails"] as? [Int], trails.count == 249,
              let ids = root["glyph_ids"] as? [String], ids.count == 249, Set(ids).count == 249,
              let catalogHash = root["catalog_sha256"] as? String,
              let referenceHash = root["reference_sha256"] as? String,
              let originalHash = root["original_sha256"] as? String,
              let share = root["original_share"] as? Double, share == 0.1,
              root["fill_rule"] as? String == "nonzero",
              root["reference_count"] as? Int == referenceCount,
              root["reference_visible_count"] as? Int == 56,
              root["original_count"] as? Int == originalCount,
              root["original_offset"] as? Int == originalOffset,
              root["blank_index"] as? Int == blankIndex,
              commands[blankIndex].isEmpty,
              speeds.allSatisfy({ $0.isFinite && $0 > 0 }),
              trails.allSatisfy({ $0 > 0 }) else { return nil }
        let paths = commands.compactMap { makePath($0) }
        guard paths.count == 249 else { return nil }
        return Catalog(commands: commands, paths: paths, speeds: speeds, trails: trails,
                       ids: ids, catalogHash: catalogHash, referenceHash: referenceHash,
                       originalHash: originalHash, originalShare: share)
    }

    static func makePath(_ commands: [[Double]]) -> CGPath? {
        let path = CGMutablePath()
        var open = false
        for c in commands {
            guard let op = c.first, op.isFinite, op.rounded() == op, (0...3).contains(op),
                  c.count == (op == 2 ? 7 : op == 3 ? 1 : 3), c.allSatisfy({ $0.isFinite }) else { return nil }
            if op == 0 {
                guard !open else { return nil }
                path.move(to: CGPoint(x: c[1], y: c[2])); open = true
            } else {
                guard open else { return nil }
                if op == 1 { path.addLine(to: CGPoint(x: c[1], y: c[2])) }
                else if op == 2 {
                    path.addCurve(to: CGPoint(x: c[5], y: c[6]),
                                  control1: CGPoint(x: c[1], y: c[2]),
                                  control2: CGPoint(x: c[3], y: c[4]))
                } else { path.closeSubpath(); open = false }
            }
        }
        return open ? nil : path.copy()
    }

    static func mix(_ input: UInt32) -> UInt32 {
        var value = input
        value ^= value >> 16; value = value &* 0x7feb352d
        value ^= value >> 15; value = value &* 0x846ca68b
        return value ^ (value >> 16)
    }

    func select(seed: UInt32, row: Int, epoch: Int) -> Int {
        let hash = Catalog.mix(seed ^ (UInt32(truncatingIfNeeded: row) &* 0x9e3779b9)
                              ^ (UInt32(truncatingIfNeeded: epoch) &* 0x85ebca6b))
        let index = Catalog.mix(hash ^ 0xa511e9b3)
        return hash % 10000 < UInt32(originalShare * 10000)
            ? Catalog.originalOffset + Int(index % UInt32(Catalog.originalCount))
            : Int(index % UInt32(Catalog.referenceCount))
    }
}

private final class Column {
    var x: CGFloat = 0
    var y: CGFloat = 0
    var glyph = 0
    var phase = 0
    var seed: UInt32 = 0
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
    var columns: [Column] = []
    var drawnReference = 0
    var drawnOriginal = 0
    var drawnBlank = 0

    init(catalog: Catalog, cell: CGFloat, spacing: CGFloat, speed: CGFloat,
         level: CGFloat, width: CGFloat, height: CGFloat) {
        self.speed = speed
        pad = max(3, cell / 2)
        trailSprites = catalog.paths.indices.map { Layer.sprite(catalog: catalog, glyph: $0, cell: cell, pad: max(3, cell / 2), level: level, head: false) }
        headSprites = catalog.paths.indices.map { Layer.sprite(catalog: catalog, glyph: $0, cell: cell, pad: max(3, cell / 2), level: level, head: true) }
        step = max(3, cell * 1.04)
        let lane = max(3, cell * spacing)
        let count = Int((width / lane).rounded(.up) + 1)
        for index in 0..<count {
            let column = Column()
            column.x = CGFloat(index) * lane + CGFloat.random(in: -3...3)
            reset(column, catalog: catalog, initial: true)
            column.y = CGFloat.random(in: 0...1) * (height + CGFloat(catalog.trails[column.glyph]) * step)
            columns.append(column)
        }
    }

    func reset(_ column: Column, catalog: Catalog, initial: Bool) {
        column.seed = UInt32.random(in: 0...UInt32.max)
        column.glyph = catalog.select(seed: column.seed, row: 0, epoch: 0)
        column.phase = Int.random(in: 0..<1000)
        column.rate = CGFloat(catalog.speeds[column.glyph]) * 5.6 * speed
        column.accumulator = CGFloat.random(in: 0...1)
        column.burst = !initial && Double.random(in: 0...1) < 0.06 ? 1.6 : 0
        if !initial { column.y = -step * CGFloat(Int.random(in: 0..<7)) }
    }

    static func sprite(catalog: Catalog, glyph: Int, cell: CGFloat,
                       pad: CGFloat, level: CGFloat, head: Bool, glow: Bool = true) -> NSImage {
        let size = cell + pad * 2
        let image = NSImage(size: NSSize(width: size, height: size))
        image.lockFocus()
        if let context = NSGraphicsContext.current?.cgContext {
            context.translateBy(x: 0, y: size)
            context.scaleBy(x: 1, y: -1)
            if glow {
                context.setShadow(offset: .zero, blur: cell * (head ? 0.14 : 0.08),
                                  color: CGColor(red: 117 / 255, green: 240 / 255, blue: 152 / 255,
                                                 alpha: (head ? 0.5 : 0.24) * level))
            }
            let lightness = 0.7 * level
            let chroma = (1 - abs(2 * lightness - 1)) * 0.8
            let hue: CGFloat = 137.0 / 60
            let secondary = chroma * (1 - abs(hue.truncatingRemainder(dividingBy: 2) - 1))
            let m = lightness - chroma / 2
            context.setFillColor(head
                ? CGColor(red: 162 / 255 * level, green: level, blue: 216 / 255 * level, alpha: 1)
                : CGColor(red: m, green: chroma + m, blue: secondary + m, alpha: 1))
            if !glow { context.setFillColor(CGColor(gray: 1, alpha: 1)) }
            context.translateBy(x: pad, y: pad)
            context.scaleBy(x: cell / 100, y: cell / 100)
            context.addPath(catalog.paths[glyph])
            // Preserve every contour and its authored direction. Even-odd or
            // one-fill-per-contour would change holes and overlapping shapes.
            context.fillPath(using: .winding)
        }
        image.unlockFocus()
        return image
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
    public override func startAnimation() { super.startAnimation(); buildScene() }
    public override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        if catalog != nil { buildScene() }
    }
    private func buildScene() {
        guard let catalog = catalog ?? Catalog.load() else { return }
        self.catalog = catalog
        let width = bounds.width, height = bounds.height
        guard width > 0, height > 0 else { return }
        let scale = max(0.5, min(1.5, height / 720))
        layers = [
            Layer(catalog: catalog, cell: 11 * scale, spacing: 1.20, speed: 0.62, level: 0.48, width: width, height: height),
            Layer(catalog: catalog, cell: 20 * scale, spacing: 1.35, speed: 0.80, level: 0.75, width: width, height: height),
            Layer(catalog: catalog, cell: 36 * scale, spacing: 2.50, speed: 1.00, level: 1.00, width: width, height: height)
        ]
        let image = NSImage(size: bounds.size)
        buffer = image
        drawFrame(catalog: catalog, buffer: image, dt: 0)
        lastTick = Date()
    }
    public override func animateOneFrame() {
        guard let catalog = catalog, let buffer = buffer else { buildScene(); needsDisplay = true; return }
        let now = Date()
        let dt = CGFloat(min(0.05, now.timeIntervalSince(lastTick)))
        lastTick = now
        drawFrame(catalog: catalog, buffer: buffer, dt: dt)
        needsDisplay = true
    }
    private func drawFrame(catalog: Catalog, buffer: NSImage, dt: CGFloat) {
        let height = bounds.height
        buffer.lockFocus()
        NSColor.black.setFill(); bounds.fill(using: .copy)
        for layer in layers {
            for column in layer.columns {
                column.accumulator += dt * column.rate * (column.burst > 0 ? 1.9 : 1)
                if column.burst > 0 { column.burst -= dt }
                while column.accumulator >= 1 {
                    column.accumulator -= 1; column.y += layer.step; column.phase += 1
                    if column.y - CGFloat(catalog.trails[column.glyph]) * layer.step * 1.15 > height,
                       Double.random(in: 0...1) < 0.6 { layer.reset(column, catalog: catalog, initial: false) }
                }
                let length = Int((Double(catalog.trails[column.glyph]) * 1.15).rounded())
                for tail in stride(from: length, through: 0, by: -1) {
                    let y = column.y - CGFloat(tail) * layer.step
                    if y < -layer.step || y > height + layer.step { continue }
                    let glyph = catalog.select(seed: column.seed, row: Int(y / layer.step), epoch: column.phase / 6)
                    if glyph == Catalog.blankIndex { layer.drawnBlank += 1; continue }
                    if glyph < Catalog.originalOffset { layer.drawnReference += 1 } else { layer.drawnOriginal += 1 }
                    let near = 1 - CGFloat(tail) / CGFloat(length)
                    let shimmer = 0.75 + CGFloat(glyph % 5) * 0.0625
                    let alpha: CGFloat = tail == 0 ? 1 : (0.25 + 0.75 * sqrt(near)) * shimmer
                    let sprite = tail == 0 ? layer.headSprites[glyph] : layer.trailSprites[glyph]
                    sprite.draw(at: NSPoint(x: column.x - layer.pad, y: height - y + layer.pad - sprite.size.height),
                                from: .zero, operation: .sourceOver, fraction: alpha)
                }
            }
        }
        buffer.unlockFocus()
    }
    public override func draw(_ rect: NSRect) {
        NSColor.black.setFill(); rect.fill()
        buffer?.draw(in: bounds, from: .zero, operation: .sourceOver, fraction: 1)
    }
    public override var hasConfigureSheet: Bool { false }
    public override var configureSheet: NSWindow? { nil }

    // These selectors are exercised by the independent compiled Objective-C
    // smoke host, after loading the packaged .saver through NSBundle.
    @objc public func catalogUsage() -> NSDictionary {
        return ["reference": layers.reduce(0) { $0 + $1.drawnReference },
                "original": layers.reduce(0) { $0 + $1.drawnOriginal },
                "blank": layers.reduce(0) { $0 + $1.drawnBlank }]
    }
    @objc public func catalogDiagnostics() -> NSDictionary {
        guard let catalog = catalog ?? Catalog.load() else { return ["status": "failed"] }
        var visible = 0, referenceCounters = 0, originalCounters = 0, curves = 0
        for index in catalog.paths.indices {
            curves += catalog.commands[index].filter { $0[0] == 2 }.count
            guard let metrics = GlyphRainView.maskMetrics(catalog.paths[index]) else { return ["status": "failed"] }
            if index == Catalog.blankIndex {
                if metrics.ink != 0 { return ["status": "failed"] }
            } else {
                if metrics.ink == 0 { return ["status": "failed"] }
                visible += 1
                if metrics.holes > 0 {
                    if index < Catalog.originalOffset { referenceCounters += 1 } else { originalCounters += 1 }
                }
            }
        }
        var originalSelections = 0, blankSelections = 0
        for row in 0..<100000 {
            let glyph = catalog.select(seed: 7319, row: row, epoch: 0)
            if glyph >= Catalog.originalOffset { originalSelections += 1 }
            if glyph == Catalog.blankIndex { blankSelections += 1 }
        }
        let passed = visible == 248 && referenceCounters > 0 && originalCounters > 0 && curves > 0
            && (9500...10500).contains(originalSelections) && blankSelections > 0
        return ["status": passed ? "passed" : "failed", "count": 249, "visible_count": visible,
                "reference_sequence_count": 57, "original_count": 192, "blank_index": 4,
                "first_original_id": catalog.ids[57], "fill_rule": "nonzero", "cubic_commands": curves,
                "reference_glyphs_with_counters_at_64px": referenceCounters,
                "original_glyphs_with_counters_at_64px": originalCounters,
                "selection_samples": 100000, "original_selections": originalSelections,
                "blank_selections": blankSelections, "catalog_sha256": catalog.catalogHash,
                "reference_sha256": catalog.referenceHash, "original_sha256": catalog.originalHash]
    }
    @objc public func catalogAtlas() -> NSImage? {
        guard let catalog = catalog ?? Catalog.load() else { return nil }
        let columns = 16, cell = 64
        let rows = (catalog.paths.count + columns - 1) / columns
        let image = NSImage(size: NSSize(width: CGFloat(columns * cell), height: CGFloat(rows * cell)))
        image.lockFocus(); NSColor.black.setFill(); NSRect(origin: .zero, size: image.size).fill()
        for index in catalog.paths.indices {
            let sprite = Layer.sprite(catalog: catalog, glyph: index, cell: CGFloat(cell), pad: 0, level: 1, head: false, glow: false)
            sprite.draw(at: NSPoint(x: CGFloat(index % columns * cell), y: CGFloat((rows - 1 - index / columns) * cell)),
                        from: .zero, operation: .sourceOver, fraction: 1)
        }
        image.unlockFocus(); return image
    }
    private static func maskMetrics(_ path: CGPath) -> (ink: Int, holes: Int)? {
        let side = 64, count = side * side
        var pixels = [UInt8](repeating: 0, count: count * 4)
        let drawn = pixels.withUnsafeMutableBytes { memory -> Bool in
            guard let context = CGContext(data: memory.baseAddress, width: side, height: side,
                bitsPerComponent: 8, bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            context.scaleBy(x: CGFloat(side) / 100, y: CGFloat(side) / 100)
            context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
            context.addPath(path); context.fillPath(using: .winding)
            return true
        }
        guard drawn else { return nil }
        let filled = (0..<count).map { pixels[$0 * 4 + 3] >= 128 }
        var visited = [Bool](repeating: false, count: count), holes = 0
        for start in 0..<count where !filled[start] && !visited[start] {
            var queue = [start], cursor = 0, edge = false
            visited[start] = true
            while cursor < queue.count {
                let p = queue[cursor], x = p % side, y = p / side
                cursor += 1
                if x == 0 || y == 0 || x == side - 1 || y == side - 1 { edge = true }
                let neighbors = [x > 0 ? p - 1 : -1, x < side - 1 ? p + 1 : -1,
                                 y > 0 ? p - side : -1, y < side - 1 ? p + side : -1]
                for n in neighbors where n >= 0 && !filled[n] && !visited[n] {
                    visited[n] = true; queue.append(n)
                }
            }
            if !edge && queue.count >= 4 { holes += 1 }
        }
        return (filled.filter { $0 }.count, holes)
    }
}
