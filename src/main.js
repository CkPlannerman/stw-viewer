/**
 * Gaussian Splat Viewer — Optimized for Proposal Presentations
 *
 * Performance optimizations:
 * - Sort worker keeps positions in memory (no 240MB copy per sort)
 * - Sort throttled to max every 200ms
 * - Render at adaptive resolution
 * - Efficient texture-based GPU rendering
 *
 * Presentation features:
 * - Camera preset tour with smooth transitions
 * - Touch controls (pinch-zoom, two-finger pan)
 * - Fullscreen mode
 * - Cinematic auto-orbit with zoom/elevation oscillation
 * - Radial morph animation (points → splats)
 * - HDR cubemap sky background
 * - 360 panorama waypoints (click to enter immersive pano)
 */

import { PanoWaypointSystem } from "./pano-viewer.js";

// ============================================================
// Shader sources
// ============================================================

const SKY_VERT = `#version 300 es
precision highp float;
in vec2 a_pos;
out vec2 v_ndc;
void main() {
    v_ndc = a_pos;
    gl_Position = vec4(a_pos, 0.9999, 1.0);
}
`;

const SKY_FRAG = `#version 300 es
precision highp float;
in vec2 v_ndc;
out vec4 fragColor;

uniform mat4 u_invViewProj;
uniform samplerCube u_envCube;

void main() {
    vec4 nearPoint = u_invViewProj * vec4(v_ndc, -1.0, 1.0);
    vec4 farPoint  = u_invViewProj * vec4(v_ndc,  1.0, 1.0);
    vec3 rayDir = normalize(farPoint.xyz / farPoint.w - nearPoint.xyz / nearPoint.w);
    vec3 color = texture(u_envCube, rayDir).rgb;
    fragColor = vec4(color, 1.0);
}
`;

const VERT_SRC = `#version 300 es
precision highp float;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform vec2 u_viewport;
uniform float u_morphProgress;
uniform float u_splatScale;
uniform float u_radialProgress;
uniform sampler2D u_posTex;
uniform sampler2D u_scaleTex;
uniform sampler2D u_colorTex;
uniform sampler2D u_rotTex;
uniform int u_texWidth;

in uint a_sortIndex;
in vec2 a_quadOffset;

out vec4 v_color;
out vec2 v_offset;
out float v_localMorph;

mat3 quatToMat3(vec4 q) {
    float x = q.y, y = q.z, z = q.w, w = q.x;
    return mat3(
        1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z),       2.0*(x*z - w*y),
        2.0*(x*y - w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x),
        2.0*(x*z + w*y),       2.0*(y*z - w*x),       1.0 - 2.0*(x*x + y*y)
    );
}

void main() {
    int idx = int(a_sortIndex);
    ivec2 texCoord = ivec2(idx % u_texWidth, idx / u_texWidth);
    vec3 a_position = texelFetch(u_posTex, texCoord, 0).xyz;
    vec3 a_scale = texelFetch(u_scaleTex, texCoord, 0).xyz;
    vec4 a_color = texelFetch(u_colorTex, texCoord, 0);
    vec4 a_rotation = texelFetch(u_rotTex, texCoord, 0);

    vec4 camPos = u_view * vec4(a_position, 1.0);

    if (camPos.z > -0.5) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        v_color = vec4(0.0);
        return;
    }

    vec4 clipPos = u_projection * camPos;
    vec2 ndc = clipPos.xy / clipPos.w;
    float screenDist = length(ndc);

    float localMorph = clamp(u_radialProgress - screenDist * 0.7, 0.0, 1.0);
    localMorph = localMorph * localMorph * (3.0 - 2.0 * localMorph);

    float minScale = 0.001;
    vec3 morphedScale = mix(vec3(minScale), a_scale * u_splatScale, localMorph);

    mat3 R = quatToMat3(a_rotation);
    mat3 S = mat3(
        morphedScale.x, 0.0, 0.0,
        0.0, morphedScale.y, 0.0,
        0.0, 0.0, morphedScale.z
    );
    mat3 M = R * S;
    mat3 Sigma = M * transpose(M);

    float tanFovY = 1.0 / u_projection[1][1];
    float tanFovX = tanFovY * u_viewport.x / u_viewport.y;

    float nx = camPos.x / camPos.z;
    float ny = camPos.y / camPos.z;
    if (abs(nx) > tanFovX * 1.5 || abs(ny) > tanFovY * 1.5) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        v_color = vec4(0.0);
        return;
    }
    camPos.x = clamp(nx, -tanFovX, tanFovX) * camPos.z;
    camPos.y = clamp(ny, -tanFovY, tanFovY) * camPos.z;

    float focal_y = u_viewport.y / (2.0 * tanFovY);
    float focal_x = u_viewport.x / (2.0 * tanFovX);

    mat3 J = mat3(
        focal_x / camPos.z, 0.0, -(focal_x * camPos.x) / (camPos.z * camPos.z),
        0.0, focal_y / camPos.z, -(focal_y * camPos.y) / (camPos.z * camPos.z),
        0.0, 0.0, 0.0
    );

    mat3 viewRot = mat3(u_view);
    mat3 T = J * viewRot;
    mat3 cov2d = T * Sigma * transpose(T);

    float a = cov2d[0][0] + 0.3;
    float b = cov2d[0][1];
    float c = cov2d[1][1] + 0.3;

    float det = a * c - b * b;
    float trace = a + c;
    float disc = max(0.01, trace * trace / 4.0 - det);
    float sqrtDisc = sqrt(disc);
    float lambda1 = trace / 2.0 + sqrtDisc;
    float lambda2 = max(0.1, trace / 2.0 - sqrtDisc);

    float r1 = sqrt(lambda1);
    float r2 = sqrt(lambda2);

    float maxRadius = 2.0 * max(r1, r2);
    maxRadius = min(maxRadius, 100.0);

    float pointSize = mix(2.0, 0.0, localMorph);
    maxRadius = max(maxRadius, pointSize);

    vec2 v1;
    if (abs(b) > 0.0001) {
        v1 = normalize(vec2(lambda1 - c, b));
    } else {
        v1 = vec2(1.0, 0.0);
    }
    vec2 v2 = vec2(-v1.y, v1.x);

    float maxScreenR = max(r1, r2) * 2.0 / max(u_viewport.x, u_viewport.y) * 2.0;
    if (maxScreenR < 0.001) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        v_color = vec4(0.0);
        return;
    }

    vec2 screenOffset = (a_quadOffset.x * v1 * r1 + a_quadOffset.y * v2 * r2) * 2.0;

    vec4 projPos = u_projection * camPos;
    vec2 ndcOffset = screenOffset * 2.0 / u_viewport;

    gl_Position = vec4(
        projPos.x / projPos.w + ndcOffset.x,
        projPos.y / projPos.w + ndcOffset.y,
        projPos.z / projPos.w,
        1.0
    );

    float dist = -camPos.z;
    float nearFade = smoothstep(0.5, 3.0, dist);

    v_color = vec4(a_color.rgb, a_color.a * nearFade);
    v_offset = a_quadOffset * 2.0;
    v_localMorph = localMorph;
}
`;

const FRAG_SRC = `#version 300 es
precision highp float;

in vec4 v_color;
in vec2 v_offset;
in float v_localMorph;

out vec4 fragColor;

void main() {
    float d2 = dot(v_offset, v_offset);
    float sharpness = mix(10.0, 1.2, v_localMorph);
    float gaussian = exp(-0.5 * sharpness * d2);

    if (gaussian < 0.02) discard;

    float splatOpacity = mix(1.0, 0.95, v_localMorph);
    float alpha = gaussian * splatOpacity * v_color.a;

    fragColor = vec4(v_color.rgb * alpha, alpha);
}
`;


// ============================================================
// Splat file loader (supports single file and chunked)
// ============================================================

async function fetchWithProgress(url, loadingBar, loadingText, label, offsetBytes, totalBytes) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status} fetching ${url}`);
    const contentLength = +response.headers.get("Content-Length") || 0;
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        if (totalBytes > 0) {
            const pct = ((offsetBytes + received) / totalBytes * 100).toFixed(0);
            loadingBar.style.width = `${pct}%`;
            loadingText.textContent = `${label} ${pct}%`;
        }
    }

    const result = new Uint8Array(received);
    let off = 0;
    for (const chunk of chunks) {
        result.set(chunk, off);
        off += chunk.length;
    }
    return result;
}

async function loadSplatFile(urlOrFile) {
    const loadingBar = document.getElementById("loading-bar");
    const loadingText = document.getElementById("loading-text");

    if (urlOrFile instanceof File) {
        loadingText.textContent = `Loading ${urlOrFile.name}...`;
        const buffer = await urlOrFile.arrayBuffer();
        loadingBar.style.width = "100%";
        return parseSplatBuffer(buffer);
    }

    loadingText.textContent = "Loading splat data...";
    try {
        const headResp = await fetch(urlOrFile, { method: "HEAD" });
        if (headResp.ok) {
            const data = await fetchWithProgress(urlOrFile, loadingBar, loadingText,
                "Loading splat data...", 0, +headResp.headers.get("Content-Length") || 0);
            return parseSplatBuffer(data.buffer);
        }
    } catch (e) { /* fall through to chunked */ }

    // Chunked loading: url.000, url.001, ...
    loadingText.textContent = "Loading splat chunks...";
    const chunkSizes = [];
    for (let i = 0; i < 100; i++) {
        const chunkUrl = `${urlOrFile}.${String(i).padStart(3, '0')}`;
        try {
            const head = await fetch(chunkUrl, { method: "HEAD" });
            if (!head.ok) break;
            chunkSizes.push({ url: chunkUrl, size: +head.headers.get("Content-Length") || 0 });
        } catch (e) { break; }
    }

    if (chunkSizes.length === 0) {
        throw new Error("Could not load splat data (no file or chunks found)");
    }

    const grandTotal = chunkSizes.reduce((s, c) => s + c.size, 0);
    const chunkBuffers = [];
    let loaded = 0;

    for (let i = 0; i < chunkSizes.length; i++) {
        const { url, size } = chunkSizes[i];
        const data = await fetchWithProgress(url, loadingBar, loadingText,
            `Loading chunk ${i + 1}/${chunkSizes.length}...`, loaded, grandTotal);
        chunkBuffers.push(data);
        loaded += data.length;
    }

    const combined = new Uint8Array(loaded);
    let offset = 0;
    for (const buf of chunkBuffers) {
        combined.set(buf, offset);
        offset += buf.length;
    }

    return parseSplatBuffer(combined.buffer);
}


function parseSplatBuffer(buffer) {
    const bytesPerSplat = 32;
    const count = Math.floor(buffer.byteLength / bytesPerSplat);

    console.log(`Parsed ${count.toLocaleString()} splats from ${(buffer.byteLength / 1024 / 1024).toFixed(1)} MB`);

    const f32 = new Float32Array(buffer);
    const u8 = new Uint8Array(buffer);

    const positions = new Float32Array(count * 3);
    const scales = new Float32Array(count * 3);
    const colors = new Float32Array(count * 4);
    const rotations = new Float32Array(count * 4);

    for (let i = 0; i < count; i++) {
        const fOff = i * 8;
        const bOff = i * 32;

        positions[i * 3 + 0] = f32[fOff + 0];
        positions[i * 3 + 1] = f32[fOff + 1];
        positions[i * 3 + 2] = f32[fOff + 2];

        scales[i * 3 + 0] = f32[fOff + 3];
        scales[i * 3 + 1] = f32[fOff + 4];
        scales[i * 3 + 2] = f32[fOff + 5];

        colors[i * 4 + 0] = u8[bOff + 24] / 255;
        colors[i * 4 + 1] = u8[bOff + 25] / 255;
        colors[i * 4 + 2] = u8[bOff + 26] / 255;
        colors[i * 4 + 3] = u8[bOff + 27] / 255;

        rotations[i * 4 + 0] = (u8[bOff + 28] - 128) / 128;
        rotations[i * 4 + 1] = (u8[bOff + 29] - 128) / 128;
        rotations[i * 4 + 2] = (u8[bOff + 30] - 128) / 128;
        rotations[i * 4 + 3] = (u8[bOff + 31] - 128) / 128;
    }

    return { count, positions, scales, colors, rotations };
}


// ============================================================
// Sorting (inline fallback, worker is preferred)
// ============================================================

function sortSplats(splatData, viewMatrix) {
    const { count, positions } = splatData;
    const depths = new Float32Array(count);

    const vx = viewMatrix[2], vy = viewMatrix[6], vz = viewMatrix[10], vw = viewMatrix[14];
    for (let i = 0; i < count; i++) {
        depths[i] = vx * positions[i * 3] + vy * positions[i * 3 + 1] + vz * positions[i * 3 + 2] + vw;
    }

    const maxDepth = Math.max(...depths.slice(0, Math.min(1000, count)).map(Math.abs)) * 2;
    const depthInv = 65535 / (maxDepth || 1);

    const keys = new Uint32Array(count);
    for (let i = 0; i < count; i++) {
        keys[i] = Math.max(0, Math.min(65535, ~~((maxDepth + depths[i]) * depthInv)));
    }

    const indices = new Uint32Array(count);
    for (let i = 0; i < count; i++) indices[i] = i;

    const counts0 = new Uint32Array(256);
    for (let i = 0; i < count; i++) counts0[keys[i] & 0xFF]++;
    const offsets0 = new Uint32Array(256);
    for (let i = 1; i < 256; i++) offsets0[i] = offsets0[i - 1] + counts0[i - 1];
    const temp = new Uint32Array(count);
    for (let i = 0; i < count; i++) {
        temp[offsets0[keys[indices[i]] & 0xFF]++] = indices[i];
    }

    const counts1 = new Uint32Array(256);
    for (let i = 0; i < count; i++) counts1[(keys[temp[i]] >> 8) & 0xFF]++;
    const offsets1 = new Uint32Array(256);
    for (let i = 1; i < 256; i++) offsets1[i] = offsets1[i - 1] + counts1[i - 1];
    const sorted = new Uint32Array(count);
    for (let i = 0; i < count; i++) {
        sorted[offsets1[(keys[temp[i]] >> 8) & 0xFF]++] = temp[i];
    }

    return sorted;
}


// ============================================================
// Sky renderer (cubemap from equirectangular)
// ============================================================

class SkyRenderer {
    constructor(gl) {
        this.gl = gl;
        this.ready = false;

        const vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, SKY_VERT);
        gl.compileShader(vs);

        const fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, SKY_FRAG);
        gl.compileShader(fs);

        this.program = gl.createProgram();
        gl.attachShader(this.program, vs);
        gl.attachShader(this.program, fs);
        gl.linkProgram(this.program);

        this.u_invViewProj = gl.getUniformLocation(this.program, "u_invViewProj");
        this.u_envCube = gl.getUniformLocation(this.program, "u_envCube");

        this.vao = gl.createVertexArray();
        gl.bindVertexArray(this.vao);
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
            -1, -1,  1, -1,  -1, 1,  -1,  1,  1, -1,   1, 1,
        ]), gl.STATIC_DRAW);
        const loc = gl.getAttribLocation(this.program, "a_pos");
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
        gl.bindVertexArray(null);

        const img = new Image();
        img.onload = () => {
            this.cubemap = this._equirectToCubemap(img, 1024);
            this.ready = true;
        };
        img.src = "assets/sky_env.jpg";
    }

    _equirectToCubemap(img, faceSize) {
        const gl = this.gl;
        const tmpCanvas = document.createElement("canvas");
        tmpCanvas.width = img.width;
        tmpCanvas.height = img.height;
        const ctx = tmpCanvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        const srcData = ctx.getImageData(0, 0, img.width, img.height).data;
        const srcW = img.width, srcH = img.height;

        const cubemap = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_CUBE_MAP, cubemap);

        const faces = [
            (u, v) => [ 1, -v, -u],
            (u, v) => [-1, -v,  u],
            (u, v) => [ u,  1,  v],
            (u, v) => [ u, -1, -v],
            (u, v) => [ u, -v,  1],
            (u, v) => [-u, -v, -1],
        ];

        for (let face = 0; face < 6; face++) {
            const faceData = new Uint8Array(faceSize * faceSize * 4);
            const dirFn = faces[face];

            for (let y = 0; y < faceSize; y++) {
                for (let x = 0; x < faceSize; x++) {
                    const u = 2 * (x + 0.5) / faceSize - 1;
                    const v = 2 * (y + 0.5) / faceSize - 1;
                    const dir = dirFn(u, v);
                    const len = Math.sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
                    const dx = dir[0]/len, dy = dir[1]/len, dz = dir[2]/len;
                    const phi = Math.atan2(dz, dx);
                    const theta = Math.asin(Math.max(-1, Math.min(1, dy)));
                    let eu = phi / (2 * Math.PI) + 0.5;
                    let ev = 0.5 - theta / Math.PI;

                    const sx = eu * srcW;
                    const sy = ev * srcH;
                    const sx0 = Math.floor(sx), sy0 = Math.floor(sy);
                    const fx = sx - sx0, fy = sy - sy0;
                    const sx1 = (sx0 + 1) % srcW;
                    const sy1 = Math.min(sy0 + 1, srcH - 1);

                    const i00 = (sy0 * srcW + sx0) * 4;
                    const i10 = (sy0 * srcW + sx1) * 4;
                    const i01 = (sy1 * srcW + sx0) * 4;
                    const i11 = (sy1 * srcW + sx1) * 4;

                    const dstIdx = (y * faceSize + x) * 4;
                    for (let c = 0; c < 3; c++) {
                        faceData[dstIdx + c] = Math.round(
                            srcData[i00+c]*(1-fx)*(1-fy) + srcData[i10+c]*fx*(1-fy) +
                            srcData[i01+c]*(1-fx)*fy     + srcData[i11+c]*fx*fy
                        );
                    }
                    faceData[dstIdx + 3] = 255;
                }
            }

            gl.texImage2D(
                gl.TEXTURE_CUBE_MAP_POSITIVE_X + face, 0,
                gl.RGBA, faceSize, faceSize, 0,
                gl.RGBA, gl.UNSIGNED_BYTE, faceData
            );
        }

        gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);

        return cubemap;
    }

    render(projMatrix, viewMatrix) {
        if (!this.ready) return;
        const gl = this.gl;
        gl.disable(gl.BLEND);
        gl.disable(gl.DEPTH_TEST);
        gl.depthMask(false);

        const skyView = new Float32Array(viewMatrix);
        skyView[12] = 0; skyView[13] = 0; skyView[14] = 0;
        const vp = mat4Multiply(skyView, projMatrix);
        const invVP = mat4Invert(vp);

        gl.useProgram(this.program);
        gl.uniformMatrix4fv(this.u_invViewProj, false, invVP);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_CUBE_MAP, this.cubemap);
        gl.uniform1i(this.u_envCube, 0);

        gl.bindVertexArray(this.vao);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        gl.bindVertexArray(null);
    }
}


// ============================================================
// WebGL2 splat renderer
// ============================================================

class SplatRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext("webgl2", {
            antialias: false,
            alpha: false,
            premultipliedAlpha: false,
        });
        if (!this.gl) throw new Error("WebGL2 not supported");

        this.gl.getExtension("EXT_color_buffer_float");
        this.gl.getExtension("OES_texture_float_linear");

        this.splatCount = 0;
        this.morphProgress = 0;
        this.radialProgress = 0;
        this.splatScale = 0.6;
        this.dataTexWidth = 4096;

        this.ssaoEnabled = true;

        this._initShaders();
        this._initQuadGeometry();
        this._initSSAO();
        this.sky = new SkyRenderer(this.gl);
    }

    _initShaders() {
        const gl = this.gl;
        const vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, VERT_SRC);
        gl.compileShader(vs);
        if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS))
            throw new Error("Vertex shader error: " + gl.getShaderInfoLog(vs));

        const fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, FRAG_SRC);
        gl.compileShader(fs);
        if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS))
            throw new Error("Fragment shader error: " + gl.getShaderInfoLog(fs));

        this.program = gl.createProgram();
        gl.attachShader(this.program, vs);
        gl.attachShader(this.program, fs);
        gl.linkProgram(this.program);
        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
            throw new Error("Program link error: " + gl.getProgramInfoLog(this.program));

        this.u_projection = gl.getUniformLocation(this.program, "u_projection");
        this.u_view = gl.getUniformLocation(this.program, "u_view");
        this.u_viewport = gl.getUniformLocation(this.program, "u_viewport");
        this.u_morphProgress = gl.getUniformLocation(this.program, "u_morphProgress");
        this.u_splatScale = gl.getUniformLocation(this.program, "u_splatScale");
        this.u_radialProgress = gl.getUniformLocation(this.program, "u_radialProgress");
        this.u_posTex = gl.getUniformLocation(this.program, "u_posTex");
        this.u_scaleTex = gl.getUniformLocation(this.program, "u_scaleTex");
        this.u_colorTex = gl.getUniformLocation(this.program, "u_colorTex");
        this.u_rotTex = gl.getUniformLocation(this.program, "u_rotTex");
        this.u_texWidth = gl.getUniformLocation(this.program, "u_texWidth");
        this.a_sortIndex = gl.getAttribLocation(this.program, "a_sortIndex");
        this.a_quadOffset = gl.getAttribLocation(this.program, "a_quadOffset");
    }

    _initQuadGeometry() {
        const gl = this.gl;
        this.quadVBO = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

        this.quadEBO = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.quadEBO);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 2, 1, 3]), gl.STATIC_DRAW);
    }

    _initSSAO() {
        const gl = this.gl;

        // Fullscreen quad for post-process
        this.ssaoQuadVAO = gl.createVertexArray();
        gl.bindVertexArray(this.ssaoQuadVAO);
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
            -1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1
        ]), gl.STATIC_DRAW);

        // Composite shader: multiply color by AO
        const compositeVS = `#version 300 es
        precision highp float;
        in vec2 a_pos;
        out vec2 v_uv;
        void main() {
            v_uv = a_pos * 0.5 + 0.5;
            gl_Position = vec4(a_pos, 0.0, 1.0);
        }`;

        const compositeFS = `#version 300 es
        precision highp float;
        in vec2 v_uv;
        out vec4 fragColor;
        uniform sampler2D u_color;
        uniform sampler2D u_depth;
        uniform vec2 u_resolution;
        uniform float u_near;
        uniform float u_far;

        float linearDepth(float d) {
            float z = d * 2.0 - 1.0;
            return (2.0 * u_near * u_far) / (u_far + u_near - z * (u_far - u_near));
        }

        void main() {
            vec4 color = texture(u_color, v_uv);
            float depth = texture(u_depth, v_uv).r;

            // Skip sky (depth = 1.0)
            if (depth >= 0.9999) {
                fragColor = color;
                return;
            }

            float centerZ = linearDepth(depth);
            float ao = 0.0;
            float radius = 3.0; // pixels

            // 8-sample SSAO
            const int SAMPLES = 8;
            vec2 offsets[8] = vec2[8](
                vec2(1.0, 0.0), vec2(-1.0, 0.0),
                vec2(0.0, 1.0), vec2(0.0, -1.0),
                vec2(0.7, 0.7), vec2(-0.7, 0.7),
                vec2(0.7, -0.7), vec2(-0.7, -0.7)
            );

            for (int i = 0; i < SAMPLES; i++) {
                vec2 sampleUV = v_uv + offsets[i] * radius / u_resolution;
                float sampleDepth = texture(u_depth, sampleUV).r;
                if (sampleDepth >= 0.9999) continue;
                float sampleZ = linearDepth(sampleDepth);
                float diff = centerZ - sampleZ;
                // Occluded if neighbor is closer (in front)
                ao += smoothstep(0.0, 2.0, diff) * smoothstep(8.0, 0.5, abs(diff));
            }

            ao = ao / float(SAMPLES);
            float occlusion = 1.0 - ao * 0.6; // max 60% darkening
            occlusion = clamp(occlusion, 0.4, 1.0);

            fragColor = vec4(color.rgb * occlusion, color.a);
        }`;

        const vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, compositeVS);
        gl.compileShader(vs);
        if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS))
            console.error("SSAO VS:", gl.getShaderInfoLog(vs));

        const fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, compositeFS);
        gl.compileShader(fs);
        if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS))
            console.error("SSAO FS:", gl.getShaderInfoLog(fs));

        this.ssaoProgram = gl.createProgram();
        gl.attachShader(this.ssaoProgram, vs);
        gl.attachShader(this.ssaoProgram, fs);

        const aLoc = 0;
        gl.bindAttribLocation(this.ssaoProgram, aLoc, "a_pos");
        gl.linkProgram(this.ssaoProgram);
        if (!gl.getProgramParameter(this.ssaoProgram, gl.LINK_STATUS))
            console.error("SSAO link:", gl.getProgramInfoLog(this.ssaoProgram));

        gl.enableVertexAttribArray(aLoc);
        gl.vertexAttribPointer(aLoc, 2, gl.FLOAT, false, 0, 0);
        gl.bindVertexArray(null);

        this.ssao_color = gl.getUniformLocation(this.ssaoProgram, "u_color");
        this.ssao_depth = gl.getUniformLocation(this.ssaoProgram, "u_depth");
        this.ssao_resolution = gl.getUniformLocation(this.ssaoProgram, "u_resolution");
        this.ssao_near = gl.getUniformLocation(this.ssaoProgram, "u_near");
        this.ssao_far = gl.getUniformLocation(this.ssaoProgram, "u_far");

        // FBO will be created/resized in render
        this.fbo = null;
        this.fboColor = null;
        this.fboDepth = null;
        this.fboWidth = 0;
        this.fboHeight = 0;
    }

    _ensureFBO(width, height) {
        if (this.fboWidth === width && this.fboHeight === height && this.fbo) return;

        const gl = this.gl;

        if (this.fbo) {
            gl.deleteFramebuffer(this.fbo);
            gl.deleteTexture(this.fboColor);
            gl.deleteTexture(this.fboDepth);
        }

        this.fbo = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);

        // Color attachment
        this.fboColor = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.fboColor);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.fboColor, 0);

        // Depth attachment (texture so we can read it in SSAO)
        this.fboDepth = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.fboDepth);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, width, height, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.fboDepth, 0);

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        this.fboWidth = width;
        this.fboHeight = height;
    }

    _createDataTexture(data, components, count) {
        const gl = this.gl;
        const texWidth = 4096;
        const texHeight = Math.ceil(count / texWidth);
        const padded = new Float32Array(texWidth * texHeight * 4);
        for (let i = 0; i < count; i++) {
            const srcOff = i * components;
            const dstOff = i * 4;
            for (let c = 0; c < components; c++) {
                padded[dstOff + c] = data[srcOff + c];
            }
        }
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, texWidth, texHeight, 0, gl.RGBA, gl.FLOAT, padded);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        return tex;
    }

    uploadSplatData(splatData) {
        const gl = this.gl;
        this.splatCount = splatData.count;
        this.dataTexWidth = 4096;

        this.posTex = this._createDataTexture(splatData.positions, 3, splatData.count);
        this.scaleTex = this._createDataTexture(splatData.scales, 3, splatData.count);
        this.colorTex = this._createDataTexture(splatData.colors, 4, splatData.count);
        this.rotTex = this._createDataTexture(splatData.rotations, 4, splatData.count);

        if (this.vao) gl.deleteVertexArray(this.vao);
        this.vao = gl.createVertexArray();
        gl.bindVertexArray(this.vao);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
        gl.enableVertexAttribArray(this.a_quadOffset);
        gl.vertexAttribPointer(this.a_quadOffset, 2, gl.FLOAT, false, 0, 0);

        this.sortIndexBuffer = gl.createBuffer();
        const identityIndices = new Uint32Array(splatData.count);
        for (let i = 0; i < splatData.count; i++) identityIndices[i] = i;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.sortIndexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, identityIndices, gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(this.a_sortIndex);
        gl.vertexAttribIPointer(this.a_sortIndex, 1, gl.UNSIGNED_INT, 0, 0);
        gl.vertexAttribDivisor(this.a_sortIndex, 1);

        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.quadEBO);
        gl.bindVertexArray(null);

        console.log(`Uploaded ${splatData.count.toLocaleString()} splats to GPU`);
    }

    updateSortOrder(sortedIndices) {
        const gl = this.gl;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.sortIndexBuffer);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, sortedIndices);
    }

    render(projMatrix, viewMatrix) {
        const gl = this.gl;
        const w = this.canvas.width, h = this.canvas.height;

        // Render to FBO if SSAO is enabled
        if (this.ssaoEnabled && this.splatCount > 0) {
            this._ensureFBO(w, h);
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
        }

        gl.viewport(0, 0, w, h);
        gl.clearColor(0.0, 0.0, 0.0, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        this.sky.render(projMatrix, viewMatrix);

        if (this.splatCount === 0) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            return;
        }

        // Render splats with depth writing (for SSAO depth buffer)
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.ALWAYS); // always write depth (back-to-front sorted)
        gl.depthMask(true);

        gl.useProgram(this.program);
        gl.uniformMatrix4fv(this.u_projection, false, projMatrix);
        gl.uniformMatrix4fv(this.u_view, false, viewMatrix);
        gl.uniform2f(this.u_viewport, w, h);
        gl.uniform1f(this.u_morphProgress, this.morphProgress);
        gl.uniform1f(this.u_splatScale, this.splatScale);
        gl.uniform1f(this.u_radialProgress, this.radialProgress);
        gl.uniform1i(this.u_texWidth, this.dataTexWidth);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.posTex);
        gl.uniform1i(this.u_posTex, 1);
        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, this.scaleTex);
        gl.uniform1i(this.u_scaleTex, 2);
        gl.activeTexture(gl.TEXTURE3);
        gl.bindTexture(gl.TEXTURE_2D, this.colorTex);
        gl.uniform1i(this.u_colorTex, 3);
        gl.activeTexture(gl.TEXTURE4);
        gl.bindTexture(gl.TEXTURE_2D, this.rotTex);
        gl.uniform1i(this.u_rotTex, 4);

        gl.bindVertexArray(this.vao);
        gl.drawElementsInstanced(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0, this.splatCount);
        gl.bindVertexArray(null);

        gl.disable(gl.BLEND);
        gl.depthFunc(gl.LESS);
        gl.depthMask(true);

        // SSAO composite pass
        if (this.ssaoEnabled) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.viewport(0, 0, w, h);
            gl.disable(gl.DEPTH_TEST);
            gl.disable(gl.BLEND);

            gl.useProgram(this.ssaoProgram);

            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, this.fboColor);
            gl.uniform1i(this.ssao_color, 0);

            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, this.fboDepth);
            gl.uniform1i(this.ssao_depth, 1);

            gl.uniform2f(this.ssao_resolution, w, h);
            gl.uniform1f(this.ssao_near, 0.1);
            gl.uniform1f(this.ssao_far, 10000.0);

            gl.bindVertexArray(this.ssaoQuadVAO);
            gl.drawArrays(gl.TRIANGLES, 0, 6);
            gl.bindVertexArray(null);

            gl.enable(gl.DEPTH_TEST);
        }
    }
}


// ============================================================
// Orbit camera with auto-orbit, swoop, tour, and touch
// ============================================================

class OrbitCamera {
    constructor(canvas) {
        this.canvas = canvas;
        this.distance = 10;
        this.theta = Math.PI / 4;
        this.phi = Math.PI / 4;
        this.target = [0, 0, 0];
        this.fov = 60;
        this.near = 0.1;
        this.far = 10000;

        this._dirty = true;
        this.autoOrbit = true;
        this.autoOrbitSpeed = 0.08;
        this.userInteracted = false;

        // Swoop
        this.swoopActive = false;
        this.swoopStartDist = 0;
        this.swoopEndDist = 0;
        this.swoopStartPhi = 0;
        this.swoopEndPhi = 0;
        this.swoopStartTime = 0;
        this.swoopDuration = 3000;

        // Cinematic
        this.cinematicTime = 0;
        this.baseDist = 0;
        this.basePhi = 0;

        // Idle auto-orbit
        this.idleTimeout = 5000; // ms before idle orbit starts
        this.lastInteractionTime = performance.now();
        this.idleOrbitSpeed = 0.04; // slow idle rotation

        this._initControls();
    }

    _initControls() {
        let dragging = false;
        let panning = false;
        let lastX = 0, lastY = 0;

        const onUserInteract = () => {
            this.userInteracted = true;
            this.autoOrbit = false;
            this.lastInteractionTime = performance.now();
        };

        // Mouse controls
        this.canvas.addEventListener("mousedown", (e) => {
            onUserInteract();
            if (e.button === 0) dragging = true;
            if (e.button === 2) panning = true;
            lastX = e.clientX;
            lastY = e.clientY;
        });

        window.addEventListener("mouseup", () => { dragging = false; panning = false; });

        window.addEventListener("mousemove", (e) => {
            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            lastX = e.clientX;
            lastY = e.clientY;

            if (dragging) {
                this.theta -= dx * 0.005;
                this.phi = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.phi + dy * 0.005));
                this._dirty = true;
            }

            if (panning) {
                const panSpeed = this.distance * 0.002;
                const right = this._getRight();
                const up = this._getUp();
                this.target[0] -= (dx * right[0] - dy * up[0]) * panSpeed;
                this.target[1] -= (dx * right[1] - dy * up[1]) * panSpeed;
                this.target[2] -= (dx * right[2] - dy * up[2]) * panSpeed;
                this._dirty = true;
            }
        });

        this.canvas.addEventListener("wheel", (e) => {
            e.preventDefault();
            onUserInteract();
            this.distance *= e.deltaY > 0 ? 1.1 : 0.9;
            this.distance = Math.max(0.1, Math.min(10000, this.distance));
            this._dirty = true;
        }, { passive: false });

        this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());

        // Touch controls
        let touches = [];
        let lastTouchDist = 0;
        let lastTouchMidX = 0, lastTouchMidY = 0;

        this.canvas.addEventListener("touchstart", (e) => {
            e.preventDefault();
            onUserInteract();
            touches = Array.from(e.touches);
            if (touches.length === 2) {
                const dx = touches[1].clientX - touches[0].clientX;
                const dy = touches[1].clientY - touches[0].clientY;
                lastTouchDist = Math.sqrt(dx * dx + dy * dy);
                lastTouchMidX = (touches[0].clientX + touches[1].clientX) / 2;
                lastTouchMidY = (touches[0].clientY + touches[1].clientY) / 2;
            } else if (touches.length === 1) {
                lastX = touches[0].clientX;
                lastY = touches[0].clientY;
            }
        }, { passive: false });

        this.canvas.addEventListener("touchmove", (e) => {
            e.preventDefault();
            const newTouches = Array.from(e.touches);

            if (newTouches.length === 1) {
                // Single finger: orbit
                const dx = newTouches[0].clientX - lastX;
                const dy = newTouches[0].clientY - lastY;
                lastX = newTouches[0].clientX;
                lastY = newTouches[0].clientY;
                this.theta -= dx * 0.005;
                this.phi = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.phi + dy * 0.005));
                this._dirty = true;
            } else if (newTouches.length === 2) {
                // Pinch: zoom
                const dx = newTouches[1].clientX - newTouches[0].clientX;
                const dy = newTouches[1].clientY - newTouches[0].clientY;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (lastTouchDist > 0) {
                    const scale = lastTouchDist / dist;
                    this.distance *= scale;
                    this.distance = Math.max(0.1, Math.min(10000, this.distance));
                }
                lastTouchDist = dist;

                // Two-finger drag: pan
                const midX = (newTouches[0].clientX + newTouches[1].clientX) / 2;
                const midY = (newTouches[0].clientY + newTouches[1].clientY) / 2;
                const panDx = midX - lastTouchMidX;
                const panDy = midY - lastTouchMidY;
                lastTouchMidX = midX;
                lastTouchMidY = midY;

                const panSpeed = this.distance * 0.002;
                const right = this._getRight();
                const up = this._getUp();
                this.target[0] -= (panDx * right[0] - panDy * up[0]) * panSpeed;
                this.target[1] -= (panDx * right[1] - panDy * up[1]) * panSpeed;
                this.target[2] -= (panDx * right[2] - panDy * up[2]) * panSpeed;
                this._dirty = true;
            }

            touches = newTouches;
        }, { passive: false });

        this.canvas.addEventListener("touchend", (e) => {
            touches = Array.from(e.touches);
            lastTouchDist = 0;
        });
    }

    setPresets() {} // removed — kept for compat

    update(dt) {
        // Idle auto-orbit: start slowly rotating after 20s of no interaction
        if (this.userInteracted && !this.swoopActive) {
            const idleTime = performance.now() - this.lastInteractionTime;
            if (idleTime > this.idleTimeout) {
                this.theta += this.idleOrbitSpeed * dt;
                this._dirty = true;
            }
        }

        // Swoop animation
        if (this.swoopActive) {
            const elapsed = performance.now() - this.swoopStartTime;
            let t = Math.min(1, elapsed / this.swoopDuration);
            t = 1 - Math.pow(1 - t, 3);

            this.distance = this.swoopStartDist + (this.swoopEndDist - this.swoopStartDist) * t;
            this.phi = this.swoopStartPhi + (this.swoopEndPhi - this.swoopStartPhi) * t;
            this.theta += this.autoOrbitSpeed * dt * t;

            if (elapsed >= this.swoopDuration) {
                this.swoopActive = false;
                this.baseDist = this.distance;
                this.basePhi = this.phi;
                this.cinematicTime = 0;
            }
            this._dirty = true;
        }

        // Auto-orbit with cinematic oscillation
        if (this.autoOrbit && !this.swoopActive) {
            this.cinematicTime += dt;
            this.theta += this.autoOrbitSpeed * dt;
            const zoomOsc = Math.sin(this.cinematicTime * 2 * Math.PI / 20);
            this.distance = this.baseDist * (1.0 + 0.3 * zoomOsc);
            const phiOsc = Math.sin(this.cinematicTime * 2 * Math.PI / 15);
            this.phi = this.basePhi + 0.15 * phiOsc;
            this._dirty = true;
        }
    }

    startSwoop() {
        this.swoopActive = true;
        this.swoopStartTime = performance.now();
        this.swoopStartDist = this.distance * 2.5;
        this.swoopEndDist = this.distance * 0.5;
        this.swoopStartPhi = Math.PI / 2 - 0.1;
        this.swoopEndPhi = 0.25;
        this.distance = this.swoopStartDist;
        this.phi = this.swoopStartPhi;
    }

    _getRight() {
        return [Math.cos(this.theta), 0, -Math.sin(this.theta)];
    }

    _getUp() {
        return [
            -Math.sin(this.theta) * Math.sin(this.phi),
            Math.cos(this.phi),
            -Math.cos(this.theta) * Math.sin(this.phi),
        ];
    }

    getEyePosition() {
        return [
            this.target[0] + this.distance * Math.cos(this.phi) * Math.sin(this.theta),
            this.target[1] + this.distance * Math.sin(this.phi),
            this.target[2] + this.distance * Math.cos(this.phi) * Math.cos(this.theta),
        ];
    }

    getViewMatrix() {
        return lookAt(this.getEyePosition(), this.target, [0, 1, 0]);
    }

    getProjectionMatrix() {
        const aspect = this.canvas.width / this.canvas.height;
        return perspective(this.fov * Math.PI / 180, aspect, this.near, this.far);
    }

    fitToScene(bbox) {
        const cx = (bbox.min[0] + bbox.max[0]) / 2;
        const cy = (bbox.min[1] + bbox.max[1]) / 2;
        const cz = (bbox.min[2] + bbox.max[2]) / 2;
        const dx = bbox.max[0] - bbox.min[0];
        const dy = bbox.max[1] - bbox.min[1];
        const dz = bbox.max[2] - bbox.min[2];
        const maxExtent = Math.max(dx, dy, dz);

        this.target = [cx, cy, cz];
        const medianExtent = [dx, dy, dz].sort((a, b) => a - b)[1];
        this.distance = Math.max(medianExtent * 0.8, maxExtent * 0.3);
        this.phi = 0.4;
        this.theta = Math.PI / 4;
        this._dirty = true;
    }
}


// ============================================================
// Matrix math utilities
// ============================================================

function perspective(fovRad, aspect, near, far) {
    const f = 1 / Math.tan(fovRad / 2);
    const nf = 1 / (near - far);
    return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, 2 * far * near * nf, 0,
    ]);
}

function lookAt(eye, target, up) {
    const zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
    let len = Math.sqrt(zx * zx + zy * zy + zz * zz);
    const z = [zx / len, zy / len, zz / len];

    const xx = up[1] * z[2] - up[2] * z[1];
    const xy = up[2] * z[0] - up[0] * z[2];
    const xz = up[0] * z[1] - up[1] * z[0];
    len = Math.sqrt(xx * xx + xy * xy + xz * xz);
    const x = [xx / len, xy / len, xz / len];

    const y = [
        z[1] * x[2] - z[2] * x[1],
        z[2] * x[0] - z[0] * x[2],
        z[0] * x[1] - z[1] * x[0],
    ];

    return new Float32Array([
        x[0], y[0], z[0], 0,
        x[1], y[1], z[1], 0,
        x[2], y[2], z[2], 0,
        -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]),
        -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]),
        -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]),
        1,
    ]);
}

function mat4Multiply(a, b) {
    const out = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            out[i * 4 + j] = 0;
            for (let k = 0; k < 4; k++) {
                out[i * 4 + j] += b[k * 4 + j] * a[i * 4 + k];
            }
        }
    }
    return out;
}

function mat4Invert(m) {
    const inv = new Float32Array(16);
    const te = m;
    const n11 = te[0], n21 = te[1], n31 = te[2], n41 = te[3];
    const n12 = te[4], n22 = te[5], n32 = te[6], n42 = te[7];
    const n13 = te[8], n23 = te[9], n33 = te[10], n43 = te[11];
    const n14 = te[12], n24 = te[13], n34 = te[14], n44 = te[15];

    const t11 = n23*n34*n42 - n24*n33*n42 + n24*n32*n43 - n22*n34*n43 - n23*n32*n44 + n22*n33*n44;
    const t12 = n14*n33*n42 - n13*n34*n42 - n14*n32*n43 + n12*n34*n43 + n13*n32*n44 - n12*n33*n44;
    const t13 = n13*n24*n42 - n14*n23*n42 + n14*n22*n43 - n12*n24*n43 - n13*n22*n44 + n12*n23*n44;
    const t14 = n14*n23*n32 - n13*n24*n32 - n14*n22*n33 + n12*n24*n33 + n13*n22*n34 - n12*n23*n34;
    const det = n11*t11 + n21*t12 + n31*t13 + n41*t14;
    if (det === 0) return new Float32Array(16);
    const d = 1 / det;

    inv[0] = t11*d;
    inv[1] = (n24*n33*n41 - n23*n34*n41 - n24*n31*n43 + n21*n34*n43 + n23*n31*n44 - n21*n33*n44)*d;
    inv[2] = (n22*n34*n41 - n24*n32*n41 + n24*n31*n42 - n21*n34*n42 - n22*n31*n44 + n21*n32*n44)*d;
    inv[3] = (n23*n32*n41 - n22*n33*n41 - n23*n31*n42 + n21*n33*n42 + n22*n31*n43 - n21*n32*n43)*d;
    inv[4] = t12*d;
    inv[5] = (n13*n34*n41 - n14*n33*n41 + n14*n31*n43 - n11*n34*n43 - n13*n31*n44 + n11*n33*n44)*d;
    inv[6] = (n14*n32*n41 - n12*n34*n41 - n14*n31*n42 + n11*n34*n42 + n12*n31*n44 - n11*n32*n44)*d;
    inv[7] = (n12*n33*n41 - n13*n32*n41 + n13*n31*n42 - n11*n33*n42 - n12*n31*n43 + n11*n32*n43)*d;
    inv[8] = t13*d;
    inv[9] = (n14*n23*n41 - n13*n24*n41 - n14*n21*n43 + n11*n24*n43 + n13*n21*n44 - n11*n23*n44)*d;
    inv[10] = (n12*n24*n41 - n14*n22*n41 + n14*n21*n42 - n11*n24*n42 - n12*n21*n44 + n11*n22*n44)*d;
    inv[11] = (n13*n22*n41 - n12*n23*n41 - n13*n21*n42 + n11*n23*n42 + n12*n21*n43 - n11*n22*n43)*d;
    inv[12] = t14*d;
    inv[13] = (n13*n24*n31 - n14*n23*n31 + n14*n21*n33 - n11*n24*n33 - n13*n21*n34 + n11*n23*n34)*d;
    inv[14] = (n14*n22*n31 - n12*n24*n31 - n14*n21*n32 + n11*n24*n32 + n12*n21*n34 - n11*n22*n34)*d;
    inv[15] = (n12*n23*n31 - n13*n22*n31 + n13*n21*n32 - n11*n23*n32 - n12*n21*n33 + n11*n22*n33)*d;
    return inv;
}


// ============================================================
// Bounding box
// ============================================================

function computeBBox(positions) {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    const n = positions.length / 3;
    for (let i = 0; i < n; i++) {
        const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
        if (x < min[0]) min[0] = x;
        if (y < min[1]) min[1] = y;
        if (z < min[2]) min[2] = z;
        if (x > max[0]) max[0] = x;
        if (y > max[1]) max[1] = y;
        if (z > max[2]) max[2] = z;
    }
    return { min, max };
}


// ============================================================
// Intro animation controller
// ============================================================

class IntroAnimation {
    constructor() {
        this.active = false;
        this.reversing = false;
        this.startTime = 0;
        this.swoopDuration = 3.0;
        this.morphDelay = 2.5;
        this.morphDuration = 6.0;
        this.radialProgress = 0;
    }

    start(skipDelay = false) {
        this.active = true;
        this.reversing = false;
        this.startTime = performance.now();
        this.radialProgress = 0;
        this._delay = skipDelay ? 0 : this.morphDelay;
    }

    reverse() {
        this.reversing = true;
        this.active = true;
        this.startTime = performance.now();
        this._reverseFrom = this.radialProgress;
    }

    // true when fully morphed and idle (ready for reverse)
    get isComplete() {
        return !this.active && !this.reversing && this.radialProgress >= 2.0;
    }

    // true when reversed back to 0 (ready for play)
    get isAtStart() {
        return !this.active && !this.reversing && this.radialProgress <= 0.01;
    }

    update() {
        if (!this.active) return;

        if (this.reversing) {
            const elapsed = (performance.now() - this.startTime) / 1000;
            let t = Math.min(1, elapsed / (this.morphDuration * 0.7)); // reverse is a bit faster
            t = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
            this.radialProgress = this._reverseFrom * (1 - t);

            if (t >= 1) {
                this.radialProgress = 0;
                this.active = false;
                this.reversing = false;
            }
            return;
        }

        const elapsed = (performance.now() - this.startTime) / 1000;
        const delay = this._delay !== undefined ? this._delay : this.morphDelay;
        if (elapsed > delay) {
            const morphElapsed = elapsed - delay;
            let t = Math.min(1, morphElapsed / this.morphDuration);
            t = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
            this.radialProgress = t * 2.2;
        }

        if (elapsed > delay + this.morphDuration + 0.5) {
            this.radialProgress = 2.2;
            this.active = false;
        }
    }
}


// ============================================================
// Main app
// ============================================================

async function init() {
    const canvas = document.getElementById("canvas");
    const loadingOverlay = document.getElementById("loading-overlay");
    const uiOverlay = document.getElementById("ui-overlay");
    const playBtn = document.getElementById("play-btn");
    const morphSlider = document.getElementById("morph-slider");
    const morphLabel = document.getElementById("morph-label");
    const infoEl = document.getElementById("info");
    const dropZone = document.getElementById("drop-zone");
    const fsBtn = document.getElementById("fs-btn");

    const renderScale = 0.85;
    function resize() {
        canvas.width = Math.round(window.innerWidth * devicePixelRatio * renderScale);
        canvas.height = Math.round(window.innerHeight * devicePixelRatio * renderScale);
    }
    resize();
    window.addEventListener("resize", resize);

    const renderer = new SplatRenderer(canvas);
    const camera = new OrbitCamera(canvas);
    const intro = new IntroAnimation();
    const panoSystem = new PanoWaypointSystem(renderer.gl, canvas);

    let rawSplatData = null;
    let sortPending = false;
    let lastSortTime = 0;
    let lastSortTheta = 0;
    let lastSortPhi = 0;
    let lastFrameTime = performance.now();
    const SORT_INTERVAL = 200; // ms — throttle sorts

    // Sort worker — positions stored in worker, only view matrix sent per sort
    const sortWorker = new Worker("src/sort-worker.js");
    sortWorker.onmessage = function(e) {
        if (!rawSplatData) return;
        renderer.updateSortOrder(e.data.sortedIndices);
        sortPending = false;
    };

    async function loadAndDisplay(source) {
        loadingOverlay.classList.remove("hidden");
        uiOverlay.style.display = "none";

        try {
            rawSplatData = await loadSplatFile(source);

            const bbox = computeBBox(rawSplatData.positions);
            camera.fitToScene(bbox);
            camera.setPresets(bbox);

            renderer.uploadSplatData(rawSplatData);

            // Send positions to worker once (stored permanently)
            sortWorker.postMessage({
                type: "init",
                positions: rawSplatData.positions,
                count: rawSplatData.count,
            });

            // Initial sort
            const viewMatrix = camera.getViewMatrix();
            const sorted = sortSplats(rawSplatData, viewMatrix);
            renderer.updateSortOrder(sorted);
            lastSortTheta = camera.theta;
            lastSortPhi = camera.phi;
            lastSortTime = performance.now();

            infoEl.textContent = `${rawSplatData.count.toLocaleString()} splats`;

            loadingOverlay.classList.add("hidden");
            uiOverlay.style.display = "flex";

            camera.startSwoop();
            intro.start();

        } catch (err) {
            console.error(err);
            document.getElementById("loading-text").textContent = "Error: " + err.message;
        }
    }

    // Drag and drop
    window.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("active"); });
    window.addEventListener("dragleave", () => { dropZone.classList.remove("active"); });
    window.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("active");
        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith(".splat")) loadAndDisplay(file);
    });

    // UI controls
    playBtn.addEventListener("click", () => {
        if (intro.isComplete) {
            intro.reverse();
        } else if (intro.isAtStart || (!intro.active && intro.radialProgress <= 0.01)) {
            intro.start(true); // skip delay on manual trigger
        }
    });

    morphSlider.addEventListener("input", () => {
        intro.active = false;
        intro.radialProgress = (morphSlider.value / 100) * 2.2;
    });

    // Waypoint click handler
    canvas.addEventListener("click", async (e) => {
        if (panoSystem.active || panoSystem.transitioning) return;

        const proj = camera.getProjectionMatrix();
        const view = camera.getViewMatrix();
        const hit = panoSystem.hitTest(proj, view, e.clientX, e.clientY);
        if (hit) {
            camera.userInteracted = true;
            camera.autoOrbit = false;
            camera.tourActive = false;
            await panoSystem.enterPanorama(hit.id, camera);
        }
    });

    // Pano mode: mouse drag for look-around
    let panoDragging = false;
    let panoLastX = 0, panoLastY = 0;
    canvas.addEventListener("mousedown", (e) => {
        if (panoSystem.active) {
            panoDragging = true;
            panoLastX = e.clientX;
            panoLastY = e.clientY;
        }
    });
    window.addEventListener("mouseup", () => { panoDragging = false; });
    window.addEventListener("mousemove", (e) => {
        if (panoDragging && panoSystem.active) {
            panoSystem.handlePanoInput(e.clientX - panoLastX, e.clientY - panoLastY, 0);
            panoLastX = e.clientX;
            panoLastY = e.clientY;
        }
    });
    canvas.addEventListener("wheel", (e) => {
        if (panoSystem.active) {
            e.preventDefault();
            panoSystem.handlePanoInput(0, 0, e.deltaY);
        }
    }, { passive: false });

    // Pano touch controls
    let panoTouchLastX = 0, panoTouchLastY = 0;
    canvas.addEventListener("touchstart", (e) => {
        if (panoSystem.active && e.touches.length === 1) {
            panoTouchLastX = e.touches[0].clientX;
            panoTouchLastY = e.touches[0].clientY;
        }
    });
    canvas.addEventListener("touchmove", (e) => {
        if (panoSystem.active && e.touches.length === 1) {
            e.preventDefault();
            const dx = e.touches[0].clientX - panoTouchLastX;
            const dy = e.touches[0].clientY - panoTouchLastY;
            panoSystem.handlePanoInput(dx, dy, 0);
            panoTouchLastX = e.touches[0].clientX;
            panoTouchLastY = e.touches[0].clientY;
        }
    }, { passive: false });

    // Back button / Escape to exit pano
    const backBtn = document.getElementById("back-btn");
    if (backBtn) {
        backBtn.addEventListener("click", () => panoSystem.exitPanorama(camera));
    }
    window.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && (panoSystem.active || panoSystem.transitioning)) {
            panoSystem.exitPanorama(camera);
        }
    });

    // Fullscreen button
    if (fsBtn) {
        fsBtn.addEventListener("click", () => {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen().catch(() => {});
                fsBtn.textContent = "Exit";
            } else {
                document.exitFullscreen();
                fsBtn.textContent = "Fullscreen";
            }
        });

        document.addEventListener("fullscreenchange", () => {
            fsBtn.textContent = document.fullscreenElement ? "Exit" : "Fullscreen";
        });
    }

    // Auto-load: ?url= param > STW-SCAN.splat > STW-SCAN.splat.000 (chunked) > test_scene.splat > prompt
    const params = new URLSearchParams(window.location.search);
    const splatUrl = params.get("url");
    if (splatUrl) {
        loadAndDisplay(splatUrl);
    } else {
        (async () => {
            const candidates = ["STW-SCAN.splat", "STW-SCAN.splat.000", "test_scene.splat"];
            for (const name of candidates) {
                try {
                    const r = await fetch(name, { method: "HEAD" });
                    if (r.ok) {
                        // For chunked, load the base name (loadSplatFile handles .000 probing)
                        const loadName = name.endsWith(".000") ? name.replace(".000", "") : name;
                        loadAndDisplay(loadName);
                        return;
                    }
                } catch (e) { /* next */ }
            }
            document.getElementById("loading-text").textContent = "Drop a .splat file to view";
        })();
    }

    // FPS tracking
    let fpsFrameCount = 0;
    let fpsLastTime = performance.now();
    let fpsDisplay = 0;

    // Render loop
    function frame() {
        requestAnimationFrame(frame);

        const now = performance.now();
        const dt = Math.min(0.1, (now - lastFrameTime) / 1000);
        lastFrameTime = now;

        // FPS
        fpsFrameCount++;
        if (now - fpsLastTime > 1000) {
            fpsDisplay = Math.round(fpsFrameCount * 1000 / (now - fpsLastTime));
            fpsLastTime = now;
            fpsFrameCount = 0;
            if (rawSplatData) {
                infoEl.textContent = `${rawSplatData.count.toLocaleString()} splats | ${fpsDisplay} fps`;
            }
        }

        camera.update(dt);
        intro.update();

        renderer.radialProgress = intro.radialProgress;
        renderer.morphProgress = Math.min(1, intro.radialProgress);

        const displayPct = Math.min(100, Math.round(intro.radialProgress / 2.2 * 100));
        morphSlider.value = displayPct;
        morphLabel.textContent = `${displayPct}%`;
        // Play/reverse button icon
        if (intro.active) {
            playBtn.innerHTML = "&#9646;&#9646;"; // pause while animating
        } else if (intro.isComplete) {
            playBtn.innerHTML = "&#9664;"; // reverse arrow when fully morphed
        } else {
            playBtn.innerHTML = "&#9654;"; // play arrow when at start
        }

        // Update panorama system
        panoSystem.update(camera);


        // Show/hide UI based on pano mode
        const inPano = panoSystem.active || (panoSystem.transitioning && panoSystem.panoOpacity > 0.3);
        if (backBtn) {
            if (inPano) {
                backBtn.style.display = "flex";
                requestAnimationFrame(() => { backBtn.style.opacity = "1"; });
            } else {
                backBtn.style.opacity = "0";
                // Hide after transition
                if (backBtn.style.display === "flex" && !panoSystem.transitioning) {
                    setTimeout(() => { if (!panoSystem.active) backBtn.style.display = "none"; }, 400);
                }
            }
        }
        uiOverlay.style.opacity = inPano ? "0" : "1";
        uiOverlay.style.pointerEvents = inPano ? "none" : "auto";

        // Throttled sort — max every 200ms, only when camera moves
        if (rawSplatData && !sortPending && !panoSystem.active && (now - lastSortTime > SORT_INTERVAL) &&
            (Math.abs(camera.theta - lastSortTheta) > 0.05 || Math.abs(camera.phi - lastSortPhi) > 0.05)) {
            sortPending = true;
            lastSortTheta = camera.theta;
            lastSortPhi = camera.phi;
            lastSortTime = now;
            sortWorker.postMessage({
                type: "sort",
                viewMatrix: camera.getViewMatrix(),
            });
        }

        const proj = camera.getProjectionMatrix();
        const view = camera.getViewMatrix();
        renderer.render(proj, view);

        // Render waypoint markers (on top of splats)
        panoSystem.renderMarkers(proj, view, now / 1000);

        // Render panorama overlay (crossfade on top of everything)
        panoSystem.renderPanorama(proj, view);
    }

    requestAnimationFrame(frame);
}

init();
