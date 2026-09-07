// Exercise the compiled saver through the native ScreenSaverView contract.
// Usage: smoke_macos BUNDLE RECEIPTS_DIRECTORY EXPECTED_NATIVE_CATALOG
#import <AppKit/AppKit.h>
#import <ScreenSaver/ScreenSaver.h>
#import <CommonCrypto/CommonDigest.h>
#include <limits.h>
#include <math.h>

@interface ScreenSaverView (SmytheCatalogSmoke)
- (NSDictionary *)catalogDiagnostics;
- (NSDictionary *)catalogUsage;
- (NSImage *)catalogAtlas;
@end

static NSString *SHA256(NSData *data) {
    if (!data || data.length > UINT_MAX) { return nil; }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    NSMutableString *result = [NSMutableString stringWithCapacity:64];
    for (NSUInteger i = 0; i < sizeof(digest); i++) { [result appendFormat:@"%02x", digest[i]]; }
    return result;
}

static NSDictionary *ReadJSON(NSString *path) {
    if (!path) { return nil; }
    NSData *data = [NSData dataWithContentsOfFile:path];
    if (!data) { return nil; }
    id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
    return [object isKindOfClass:NSDictionary.class] ? object : nil;
}

static BOOL HasCatalogAPI(ScreenSaverView *view) {
    return [view respondsToSelector:@selector(catalogDiagnostics)]
        && [view respondsToSelector:@selector(catalogUsage)]
        && [view respondsToSelector:@selector(catalogAtlas)];
}

static NSBitmapImageRep *Capture(ScreenSaverView *view) {
    NSBitmapImageRep *bitmap = [[NSBitmapImageRep alloc]
        initWithBitmapDataPlanes:NULL pixelsWide:(NSInteger)view.bounds.size.width
        pixelsHigh:(NSInteger)view.bounds.size.height bitsPerSample:8 samplesPerPixel:4
        hasAlpha:YES isPlanar:NO colorSpaceName:NSCalibratedRGBColorSpace
        bytesPerRow:0 bitsPerPixel:0];
    if (!bitmap) { return nil; }
    NSGraphicsContext *context = [NSGraphicsContext graphicsContextWithBitmapImageRep:bitmap];
    if (!context) { return nil; }
    [NSGraphicsContext saveGraphicsState];
    [NSGraphicsContext setCurrentContext:context];
    [view drawRect:view.bounds];
    [NSGraphicsContext restoreGraphicsState];
    return bitmap;
}

static NSDictionary *Inspect(NSBitmapImageRep *bitmap, NSBitmapImageRep *before) {
    NSUInteger samples = 0, green = 0, dark = 0, changed = 0;
    for (NSInteger y = 0; y < bitmap.pixelsHigh; y += 4) {
        for (NSInteger x = 0; x < bitmap.pixelsWide; x += 4) {
            NSColor *pixel = [[bitmap colorAtX:x y:y] colorUsingColorSpace:NSColorSpace.sRGBColorSpace];
            if (!pixel) { return nil; }
            CGFloat r = pixel.redComponent, g = pixel.greenComponent, b = pixel.blueComponent;
            samples++;
            if (g > 0.24 && g > r * 1.15 && g > b * 1.15) { green++; }
            if (fmax(r, fmax(g, b)) < 0.08) { dark++; }
            if (before) {
                NSColor *prior = [[before colorAtX:x y:y] colorUsingColorSpace:NSColorSpace.sRGBColorSpace];
                if (fabs(prior.redComponent - r) + fabs(prior.greenComponent - g)
                    + fabs(prior.blueComponent - b) > 0.025) { changed++; }
            }
        }
    }
    if (!samples) { return nil; }
    double greenFraction = (double)green / samples;
    double darkFraction = (double)dark / samples;
    double changedFraction = (double)changed / samples;
    if (greenFraction < 0.001 || greenFraction > 0.75 || darkFraction < 0.10
        || (before && changedFraction < 0.001)) {
        NSLog(@"Invalid native frame: green=%.4f dark=%.4f changed=%.4f",
              greenFraction, darkFraction, changedFraction);
        return nil;
    }
    return @{@"width": @(bitmap.pixelsWide), @"height": @(bitmap.pixelsHigh),
             @"green_fraction": @(greenFraction), @"dark_fraction": @(darkFraction),
             @"changed_fraction": @(changedFraction), @"sample_count": @(samples)};
}

static BOOL WritePNG(NSBitmapImageRep *bitmap, NSString *directory, NSString *name) {
    NSData *png = [bitmap representationUsingType:NSBitmapImageFileTypePNG properties:@{}];
    return png && [png writeToFile:[directory stringByAppendingPathComponent:name] atomically:YES];
}

static NSBitmapImageRep *AtlasBitmap(NSImage *image) {
    if (!image || image.size.width != 1024 || image.size.height != 1024) { return nil; }
    NSBitmapImageRep *bitmap = [[NSBitmapImageRep alloc]
        initWithBitmapDataPlanes:NULL pixelsWide:1024 pixelsHigh:1024
        bitsPerSample:8 samplesPerPixel:4 hasAlpha:YES isPlanar:NO
        colorSpaceName:NSCalibratedRGBColorSpace bytesPerRow:0 bitsPerPixel:0];
    NSGraphicsContext *context = [NSGraphicsContext graphicsContextWithBitmapImageRep:bitmap];
    if (!bitmap || !context) { return nil; }
    [NSGraphicsContext saveGraphicsState];
    [NSGraphicsContext setCurrentContext:context];
    [image drawInRect:NSMakeRect(0, 0, 1024, 1024) fromRect:NSZeroRect
           operation:NSCompositingOperationCopy fraction:1 respectFlipped:NO hints:nil];
    [NSGraphicsContext restoreGraphicsState];
    return bitmap;
}

static NSDictionary *InspectAtlas(NSBitmapImageRep *atlas) {
    if (!atlas || atlas.pixelsWide != 1024 || atlas.pixelsHigh != 1024) { return nil; }
    NSMutableArray *glyphs = [NSMutableArray arrayWithCapacity:249];
    NSUInteger visible = 0, referenceCounters = 0, originalCounters = 0;
    for (NSUInteger glyph = 0; glyph < 249; glyph++) {
        BOOL filled[4096] = {0}, visited[4096] = {0};
        NSUInteger ink = 0, holes = 0;
        for (NSUInteger y = 0; y < 64; y++) for (NSUInteger x = 0; x < 64; x++) {
            NSColor *pixel = [[atlas colorAtX:(NSInteger)(glyph % 16 * 64 + x)
                                          y:(NSInteger)(glyph / 16 * 64 + y)]
                colorUsingColorSpace:NSColorSpace.sRGBColorSpace];
            if (!pixel || fabs(pixel.redComponent - pixel.greenComponent) > 0.01
                || fabs(pixel.redComponent - pixel.blueComponent) > 0.01) { return nil; }
            filled[y * 64 + x] = pixel.redComponent >= 0.5;
            if (filled[y * 64 + x]) { ink++; }
        }
        for (NSUInteger start = 0; start < 4096; start++) {
            if (filled[start] || visited[start]) { continue; }
            NSUInteger queue[4096], read = 0, count = 1;
            BOOL edge = NO;
            queue[0] = start; visited[start] = YES;
            while (read < count) {
                NSUInteger p = queue[read++], x = p % 64, y = p / 64;
                if (x == 0 || x == 63 || y == 0 || y == 63) { edge = YES; }
                NSInteger neighbors[] = {x > 0 ? (NSInteger)p - 1 : -1,
                    x < 63 ? (NSInteger)p + 1 : -1, y > 0 ? (NSInteger)p - 64 : -1,
                    y < 63 ? (NSInteger)p + 64 : -1};
                for (NSUInteger i = 0; i < 4; i++) {
                    NSInteger n = neighbors[i];
                    if (n >= 0 && !filled[n] && !visited[n]) {
                        visited[n] = YES; queue[count++] = (NSUInteger)n;
                    }
                }
            }
            if (!edge && count >= 4) { holes++; }
        }
        if (glyph == 4) {
            if (ink != 0) { NSLog(@"Reference blank atlas cell is not empty"); return nil; }
        } else {
            if (ink == 0) { NSLog(@"Native atlas glyph %lu is empty", (unsigned long)glyph); return nil; }
            visible++;
            if (holes > 0) { if (glyph < 57) { referenceCounters++; } else { originalCounters++; } }
        }
        [glyphs addObject:@{@"index": @(glyph), @"ink_pixels": @(ink), @"holes_ge4px": @(holes)}];
    }
    if (visible != 248 || !referenceCounters || !originalCounters) { return nil; }
    return @{@"cell_pixels": @64, @"columns": @16, @"visible_count": @(visible),
             @"blank_index": @4, @"reference_glyphs_with_counters": @(referenceCounters),
             @"original_glyphs_with_counters": @(originalCounters), @"glyphs": glyphs,
             @"method": @"Independent white-on-black atlas threshold 0.5; four-connected counter area >=4px"};
}

static NSDictionary *CheckCatalog(Class principal, NSDictionary *resource, NSDictionary *expected,
                                  NSString *out) {
    ScreenSaverView *view = [[principal alloc] initWithFrame:NSMakeRect(0, 0, 320, 180) isPreview:YES];
    if (!view || !HasCatalogAPI(view)) { return nil; }
    NSDictionary *diagnostics = [view catalogDiagnostics];
    if (![diagnostics[@"status"] isEqual:@"passed"]
        || [diagnostics[@"count"] integerValue] != 249
        || [diagnostics[@"visible_count"] integerValue] != 248
        || [diagnostics[@"reference_sequence_count"] integerValue] != 57
        || [diagnostics[@"original_count"] integerValue] != 192
        || [diagnostics[@"blank_index"] integerValue] != 4
        || ![diagnostics[@"first_original_id"] isEqual:@"GLYPH-000"]
        || ![diagnostics[@"fill_rule"] isEqual:@"nonzero"]
        || [diagnostics[@"cubic_commands"] integerValue] <= 0
        || [diagnostics[@"reference_glyphs_with_counters_at_64px"] integerValue] <= 0
        || [diagnostics[@"original_glyphs_with_counters_at_64px"] integerValue] <= 0
        || [diagnostics[@"selection_samples"] integerValue] != 100000
        || [diagnostics[@"original_selections"] integerValue] < 9500
        || [diagnostics[@"original_selections"] integerValue] > 10500
        || [diagnostics[@"blank_selections"] integerValue] <= 0) { return nil; }
    for (NSString *key in @[@"catalog_sha256", @"reference_sha256", @"original_sha256"]) {
        if (![diagnostics[key] isEqual:resource[key]]) { return nil; }
    }
    if (![diagnostics[@"catalog_sha256"] isEqual:expected[@"catalog_sha256"]]) { return nil; }
    NSBitmapImageRep *atlas = AtlasBitmap([view catalogAtlas]);
    NSDictionary *metrics = InspectAtlas(atlas);
    if (!metrics || !WritePNG(atlas, out, @"catalog-atlas.png")) { return nil; }
    NSString *atlasHash = SHA256([NSData dataWithContentsOfFile:[out stringByAppendingPathComponent:@"catalog-atlas.png"]]);
    if (!atlasHash) { return nil; }
    return @{@"diagnostics": diagnostics, @"atlas": metrics,
             @"atlas_file": @"catalog-atlas.png", @"atlas_sha256": atlasHash};
}

static NSDictionary *CheckMode(Class principal, BOOL preview, NSSize size, NSString *out) {
    NSString *mode = preview ? @"preview" : @"fullscreen";
    ScreenSaverView *view = [[principal alloc] initWithFrame:(NSRect){NSZeroPoint, size}
                                                 isPreview:preview];
    if (!view || [view isPreview] != preview || !HasCatalogAPI(view)) { return nil; }
    [view startAnimation];
    NSDictionary *usageBefore = [view catalogUsage];
    NSBitmapImageRep *before = Capture(view);
    // The renderer uses elapsed time. A tight loop alone cannot prove motion.
    for (int frame = 0; frame < 16; frame++) {
        [NSThread sleepForTimeInterval:0.025];
        [view animateOneFrame];
    }
    NSBitmapImageRep *after = Capture(view);
    NSDictionary *animation = before && after ? Inspect(after, before) : nil;
    NSDictionary *usageAfter = [view catalogUsage];
    NSMutableDictionary *usageDelta = [NSMutableDictionary dictionary];
    for (NSString *key in @[@"reference", @"original", @"blank"]) {
        long long delta = [usageAfter[key] longLongValue] - [usageBefore[key] longLongValue];
        if (delta <= 0) { NSLog(@"No %@ glyph selections during %@ animation", key, mode); return nil; }
        usageDelta[key] = @(delta);
    }
    [view setFrameSize:NSMakeSize(size.width + 64, size.height + 36)];
    [view animateOneFrame];
    NSBitmapImageRep *resized = Capture(view);
    NSDictionary *resize = resized ? Inspect(resized, nil) : nil;
    [view stopAnimation];
    if ([view isAnimating] || !animation || !resize
        || !WritePNG(before, out, [mode stringByAppendingString:@"-initial.png"])
        || !WritePNG(after, out, [mode stringByAppendingString:@"-animated.png"])
        || !WritePNG(resized, out, [mode stringByAppendingString:@"-resized.png"])) { return nil; }
    NSLog(@"%@ passed: %@; resize: %@", mode, animation, resize);
    return @{@"mode": mode, @"is_preview": @(preview), @"status": @"passed",
             @"animation_frames": @16, @"animation": animation, @"resize": resize,
             @"catalog_draws_during_animation": usageDelta, @"stopped": @YES};
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 4) {
            NSLog(@"Usage: smoke_macos BUNDLE RECEIPTS_DIRECTORY EXPECTED_NATIVE_CATALOG");
            return 2;
        }
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyProhibited];
        NSString *path = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        NSString *out = [[NSString stringWithUTF8String:argv[2]] stringByStandardizingPath];
        NSDictionary *expected = ReadJSON([NSString stringWithUTF8String:argv[3]]);
        if (!path.isAbsolutePath) {
            path = [NSFileManager.defaultManager.currentDirectoryPath stringByAppendingPathComponent:path];
        }
        if (!out.isAbsolutePath) {
            out = [NSFileManager.defaultManager.currentDirectoryPath stringByAppendingPathComponent:out];
        }
        NSBundle *bundle = [NSBundle bundleWithPath:path];
        NSError *error = nil;
        if (![NSFileManager.defaultManager createDirectoryAtPath:out
                withIntermediateDirectories:YES attributes:nil error:&error]) {
            NSLog(@"Unable to create receipts directory: %@", error);
            return 1;
        }
        if (!bundle || ![bundle loadAndReturnError:&error]) {
            NSLog(@"Unable to load screensaver: %@", error);
            return 1;
        }
        Class principal = bundle.principalClass;
        if (!principal || ![principal isSubclassOfClass:ScreenSaverView.class] || !expected) { return 1; }
        @try {
            NSString *resourcePath = [bundle pathForResource:@"glyphs" ofType:@"json"];
            NSDictionary *resource = ReadJSON(resourcePath);
            NSString *resourceHash = SHA256([NSData dataWithContentsOfFile:resourcePath]);
            NSString *binaryHash = SHA256([NSData dataWithContentsOfFile:bundle.executablePath]);
            NSDictionary *outputHashes = expected[@"output_sha256"];
            if (!resource || !resourceHash || !binaryHash
                || ![resourceHash isEqual:outputHashes[@"screensaver/macos/glyphs.json"]]) {
                NSLog(@"Packaged native catalog does not match the shared export"); return 1;
            }
            NSDictionary *catalog = CheckCatalog(principal, resource, expected, out);
            if (!catalog) { NSLog(@"Exact SVG catalog/atlas verification failed"); return 1; }
            NSDictionary *preview = CheckMode(principal, YES, NSMakeSize(320, 180), out);
            NSDictionary *fullscreen = CheckMode(principal, NO, NSMakeSize(1280, 720), out);
            if (!preview || !fullscreen) { return 1; }
#if defined(__arm64__)
            NSString *architecture = @"arm64";
#elif defined(__x86_64__)
            NSString *architecture = @"x86_64";
#else
            NSString *architecture = @"unknown";
#endif
            NSDictionary *receipt = @{
                @"status": @"passed", @"architecture": architecture,
                @"principal_class": NSStringFromClass(principal),
                @"bundle_executable": bundle.executablePath,
                @"binary_sha256": binaryHash, @"catalog_resource_sha256": resourceHash,
                @"catalog": catalog,
                @"checks": @[preview, fullscreen],
                @"scope": @"Loaded universal bundle bytes, exact shared catalog, all248 visible SVG shapes, blank slot, nonzero counters, weighted selections, both families in normal motion, resize, and stop"
            };
            NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:&error];
            if (!json || ![json writeToFile:[out stringByAppendingPathComponent:@"macos-smoke.json"]
                                atomically:YES]) { return 1; }
        } @catch (NSException *exception) {
            NSLog(@"Native smoke failed: %@", exception);
            return 1;
        }
    }
    return 0;
}
