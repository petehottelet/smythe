// Exercise the compiled saver through the native ScreenSaverView contract.
// Usage: smoke_macos /absolute/GlyphRain.saver /absolute/receipts-directory
#import <AppKit/AppKit.h>
#import <ScreenSaver/ScreenSaver.h>
#include <math.h>

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

static NSDictionary *CheckMode(Class principal, BOOL preview, NSSize size, NSString *out) {
    NSString *mode = preview ? @"preview" : @"fullscreen";
    ScreenSaverView *view = [[principal alloc] initWithFrame:(NSRect){NSZeroPoint, size}
                                                 isPreview:preview];
    if (!view || [view isPreview] != preview) { return nil; }
    [view startAnimation];
    NSBitmapImageRep *before = Capture(view);
    // The renderer uses elapsed time. A tight loop alone cannot prove motion.
    for (int frame = 0; frame < 16; frame++) {
        [NSThread sleepForTimeInterval:0.025];
        [view animateOneFrame];
    }
    NSBitmapImageRep *after = Capture(view);
    NSDictionary *animation = before && after ? Inspect(after, before) : nil;
    [view setFrameSize:NSMakeSize(size.width + 64, size.height + 36)];
    [view animateOneFrame];
    NSBitmapImageRep *resized = Capture(view);
    NSDictionary *resize = resized ? Inspect(resized, nil) : nil;
    [view stopAnimation];
    if (!animation || !resize
        || !WritePNG(before, out, [mode stringByAppendingString:@"-initial.png"])
        || !WritePNG(after, out, [mode stringByAppendingString:@"-animated.png"])
        || !WritePNG(resized, out, [mode stringByAppendingString:@"-resized.png"])) { return nil; }
    NSLog(@"%@ passed: %@; resize: %@", mode, animation, resize);
    return @{@"mode": mode, @"is_preview": @(preview), @"status": @"passed",
             @"animation_frames": @16, @"animation": animation, @"resize": resize};
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) {
            NSLog(@"Usage: smoke_macos BUNDLE RECEIPTS_DIRECTORY");
            return 2;
        }
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyProhibited];
        NSString *path = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        NSString *out = [[NSString stringWithUTF8String:argv[2]] stringByStandardizingPath];
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
        if (!principal || ![principal isSubclassOfClass:ScreenSaverView.class]) { return 1; }
        @try {
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
                @"checks": @[preview, fullscreen],
                @"scope": @"Native bundle load, preview/fullscreen rendering, motion, resize, and stop"
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
