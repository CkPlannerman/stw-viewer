/**
 * Web Worker for back-to-front splat sorting.
 * Runs radix sort off the main thread to avoid frame drops.
 */

self.onmessage = function(e) {
    const { positions, viewMatrix, count } = e.data;

    // Compute camera-space Z for each splat
    const vx = viewMatrix[2], vy = viewMatrix[6], vz = viewMatrix[10], vw = viewMatrix[14];
    const depths = new Float32Array(count);

    for (let i = 0; i < count; i++) {
        depths[i] = vx * positions[i * 3] + vy * positions[i * 3 + 1] + vz * positions[i * 3 + 2] + vw;
    }

    // Find depth range from sample (faster than scanning all)
    let maxDepth = 0;
    const sampleStep = Math.max(1, Math.floor(count / 2000));
    for (let i = 0; i < count; i += sampleStep) {
        const abs = depths[i] < 0 ? -depths[i] : depths[i];
        if (abs > maxDepth) maxDepth = abs;
    }
    maxDepth *= 1.5;

    const depthInv = 65535 / (maxDepth || 1);

    // Convert to uint16 keys (back-to-front: far objects = small key = drawn first)
    // Camera-space Z is negative for objects in front, more negative = farther
    // So (maxDepth + depth) gives smaller values for farther objects
    const keys = new Uint32Array(count);
    for (let i = 0; i < count; i++) {
        keys[i] = Math.max(0, Math.min(65535, ~~((maxDepth + depths[i]) * depthInv)));
    }

    // Two-pass radix sort (16-bit, 8 bits per pass)
    const indices = new Uint32Array(count);
    for (let i = 0; i < count; i++) indices[i] = i;

    // Pass 1: lower 8 bits
    const counts0 = new Uint32Array(256);
    for (let i = 0; i < count; i++) counts0[keys[i] & 0xFF]++;
    const offsets0 = new Uint32Array(256);
    for (let i = 1; i < 256; i++) offsets0[i] = offsets0[i - 1] + counts0[i - 1];
    const temp = new Uint32Array(count);
    for (let i = 0; i < count; i++) {
        const k = keys[indices[i]] & 0xFF;
        temp[offsets0[k]++] = indices[i];
    }

    // Pass 2: upper 8 bits
    const counts1 = new Uint32Array(256);
    for (let i = 0; i < count; i++) counts1[(keys[temp[i]] >> 8) & 0xFF]++;
    const offsets1 = new Uint32Array(256);
    for (let i = 1; i < 256; i++) offsets1[i] = offsets1[i - 1] + counts1[i - 1];
    const sorted = new Uint32Array(count);
    for (let i = 0; i < count; i++) {
        const k = (keys[temp[i]] >> 8) & 0xFF;
        sorted[offsets1[k]++] = temp[i];
    }

    self.postMessage({ sortedIndices: sorted }, [sorted.buffer]);
};
