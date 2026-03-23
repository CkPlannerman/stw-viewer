/**
 * Panorama Waypoint System
 *
 * Renders clickable waypoint markers in 3D space. On click, flies camera
 * to the waypoint position, then crossfades into an immersive 360 panorama.
 * User can look around in the panorama, then exit back to the point cloud.
 *
 * Uses equirectangular → cubemap conversion for seamless 360 viewing.
 */

// Waypoint data: scan positions in viewer space
// Computed from RegistrationInfos.csv with centroid offset (16.99, -1.38, 103.73)
// Positions corrected for unit mismatch: registration CSV is in meters,
// LAS export is in US Survey Feet. Scale factor: 3.28084
const WAYPOINTS = [
    { id: 1, pos: [-16.99, 1.38, -103.73], file: "1.jpeg" },
    { id: 2, pos: [-17.17, 1.18, -76.98], file: "2.jpeg" },
    { id: 3, pos: [-17.11, 1.34, -49.88], file: "3.jpeg" },
    { id: 4, pos: [-43.33, 1.44, -26.21], file: "4.jpeg" },
    { id: 5, pos: [-67.01, 1.10, -26.23], file: "5.jpeg" },
    { id: 6, pos: [-80.77, 1.57, 31.69], file: "6.jpeg" },
    { id: 7, pos: [-53.06, 1.43, 31.85], file: "7.jpeg" },
    { id: 8, pos: [-29.46, 1.41, 49.54], file: "8.jpeg" },
    { id: 9, pos: [-17.19, 0.89, 73.37], file: "9.jpeg" },
    { id: 10, pos: [15.68, 1.09, 79.90], file: "10.jpeg" },
    { id: 11, pos: [17.90, 1.35, 54.67], file: "11.jpeg" },
    { id: 12, pos: [25.00, 1.46, 37.10], file: "12.jpeg" },
    { id: 13, pos: [46.58, 1.41, 31.14], file: "13.jpeg" },
    { id: 14, pos: [68.74, 1.03, 30.41], file: "14.jpeg" },
    { id: 15, pos: [89.81, 1.27, -27.68], file: "15.jpeg" },
    { id: 16, pos: [52.08, 1.17, -28.47], file: "16.jpeg" },
    { id: 17, pos: [41.87, 1.21, -29.28], file: "17.jpeg" },
    { id: 18, pos: [30.03, 1.25, -46.02], file: "18.jpeg" },
    { id: 19, pos: [16.71, 1.07, -56.20], file: "19.jpeg" },
    { id: 20, pos: [16.12, 1.42, -91.94], file: "20.jpeg" },
    { id: 21, pos: [-136.67, 1.89, 1.72], file: "21.jpeg" },
    { id: 22, pos: [-15.25, 0.86, 118.27], file: "22.jpeg" },
    { id: 23, pos: [141.44, 2.00, 2.50], file: "23.jpeg" },
];

// ============================================================
// Waypoint Marker Renderer (billboarded quads in 3D)
// ============================================================

const MARKER_VERT = `#version 300 es
precision highp float;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform vec2 u_viewport;
uniform vec3 u_markerPos;
uniform float u_pulse;      // 0-1 pulsing animation
uniform float u_hover;      // 1 if hovered
uniform float u_scale;

in vec2 a_pos;
out vec2 v_uv;
out float v_pulse;
out float v_hover;

void main() {
    vec4 center = u_projection * u_view * vec4(u_markerPos, 1.0);

    // Skip if behind camera
    if (center.w < 0.1) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    // Billboard size in pixels, with distance attenuation
    float dist = center.w;
    float baseSize = 40.0 * u_scale;
    float size = baseSize / dist;
    size = clamp(size, 8.0, 50.0); // min/max pixel size
    size *= (1.0 + 0.12 * u_pulse); // gentle pulse
    size *= (1.0 + 0.35 * u_hover); // expand on hover

    vec2 offset = a_pos * size;
    vec2 ndcOffset = offset * 2.0 / u_viewport;

    gl_Position = vec4(
        center.x / center.w + ndcOffset.x,
        center.y / center.w + ndcOffset.y,
        center.z / center.w - 0.001, // slight depth bias to render on top
        1.0
    );

    v_uv = a_pos;
    v_pulse = u_pulse;
    v_hover = u_hover;
}
`;

const MARKER_FRAG = `#version 300 es
precision highp float;

in vec2 v_uv;
in float v_pulse;
in float v_hover;
out vec4 fragColor;

void main() {
    float dist = length(v_uv);

    // Outer ring (thinner, more elegant)
    float ring = smoothstep(0.92, 0.80, dist) - smoothstep(0.72, 0.62, dist);
    // Inner dot
    float dot = smoothstep(0.30, 0.15, dist);
    // Soft glow halo
    float glow = exp(-dist * dist * 2.5) * 0.35;
    // Extra hover glow
    float hoverGlow = v_hover * exp(-dist * dist * 1.5) * 0.4;

    float alpha = max(max(ring, dot), glow) + hoverGlow;
    alpha *= (0.75 + 0.25 * v_pulse);

    // White with slight blue tint, gold-white on hover
    vec3 baseColor = vec3(0.82, 0.88, 1.0);
    vec3 hoverColor = vec3(1.0, 0.97, 0.90);
    vec3 color = mix(baseColor, hoverColor, v_hover);
    alpha = clamp(alpha, 0.0, 1.0);

    if (alpha < 0.01) discard;

    fragColor = vec4(color * alpha, alpha);
}
`;


// ============================================================
// 360 Panorama Renderer
// ============================================================

const PANO_VERT = `#version 300 es
precision highp float;
in vec2 a_pos;
out vec2 v_ndc;
void main() {
    v_ndc = a_pos;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
`;

const PANO_FRAG = `#version 300 es
precision highp float;
in vec2 v_ndc;
out vec4 fragColor;

uniform mat4 u_invViewProj;
uniform samplerCube u_panoCube;
uniform float u_opacity;

void main() {
    vec4 nearPoint = u_invViewProj * vec4(v_ndc, -1.0, 1.0);
    vec4 farPoint  = u_invViewProj * vec4(v_ndc,  1.0, 1.0);
    vec3 rayDir = normalize(farPoint.xyz / farPoint.w - nearPoint.xyz / nearPoint.w);
    vec3 color = texture(u_panoCube, rayDir).rgb;
    fragColor = vec4(color, u_opacity);
}
`;


export class PanoWaypointSystem {
    constructor(gl, canvas) {
        this.gl = gl;
        this.canvas = canvas;
        this.waypoints = WAYPOINTS;

        // State
        this.active = false;          // true when in panorama mode
        this.transitioning = false;   // true during fly-to / crossfade
        this.transitionStart = 0;
        this.transitionDuration = 1500; // ms for camera fly in
        this.fadeDuration = 600;        // ms for crossfade in
        this.exitFadeDuration = 800;    // ms for crossfade out (slower)
        this.exitCameraRestoreDuration = 1200; // ms for camera restore (smooth)
        this.currentWaypoint = null;
        this.hoveredWaypoint = null;
        this.panoOpacity = 0;

        // Camera state to restore on exit
        this._savedCamera = null;

        // Panorama camera (independent from main camera)
        this.panoTheta = 0;
        this.panoPhi = 0;
        this.panoFov = 75;

        // Loaded panorama cubemaps (cached)
        this.panoCubemaps = {};
        this.loadingPano = null;

        // Hover label element
        this._label = document.createElement("div");
        this._label.style.cssText = "position:fixed;pointer-events:none;z-index:15;background:rgba(0,0,0,0.75);color:#fff;font-size:12px;padding:4px 10px;border-radius:4px;white-space:nowrap;display:none;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;backdrop-filter:blur(6px);letter-spacing:0.3px;";
        document.body.appendChild(this._label);

        this._initMarkerRenderer();
        this._initPanoRenderer();
        this._initPickBuffer();
    }

    _initMarkerRenderer() {
        const gl = this.gl;

        const vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, MARKER_VERT);
        gl.compileShader(vs);
        if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS))
            console.error("Marker VS:", gl.getShaderInfoLog(vs));

        const fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, MARKER_FRAG);
        gl.compileShader(fs);
        if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS))
            console.error("Marker FS:", gl.getShaderInfoLog(fs));

        this.markerProgram = gl.createProgram();
        gl.attachShader(this.markerProgram, vs);
        gl.attachShader(this.markerProgram, fs);
        gl.linkProgram(this.markerProgram);

        this.m_projection = gl.getUniformLocation(this.markerProgram, "u_projection");
        this.m_view = gl.getUniformLocation(this.markerProgram, "u_view");
        this.m_viewport = gl.getUniformLocation(this.markerProgram, "u_viewport");
        this.m_markerPos = gl.getUniformLocation(this.markerProgram, "u_markerPos");
        this.m_pulse = gl.getUniformLocation(this.markerProgram, "u_pulse");
        this.m_hover = gl.getUniformLocation(this.markerProgram, "u_hover");
        this.m_scale = gl.getUniformLocation(this.markerProgram, "u_scale");

        // Quad VAO
        this.markerVAO = gl.createVertexArray();
        gl.bindVertexArray(this.markerVAO);
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
            -1, -1,  1, -1,  -1, 1,  -1, 1,  1, -1,  1, 1,
        ]), gl.STATIC_DRAW);
        const loc = gl.getAttribLocation(this.markerProgram, "a_pos");
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
        gl.bindVertexArray(null);
    }

    _initPanoRenderer() {
        const gl = this.gl;

        const vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, PANO_VERT);
        gl.compileShader(vs);

        const fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, PANO_FRAG);
        gl.compileShader(fs);

        this.panoProgram = gl.createProgram();
        gl.attachShader(this.panoProgram, vs);
        gl.attachShader(this.panoProgram, fs);
        gl.linkProgram(this.panoProgram);

        this.p_invViewProj = gl.getUniformLocation(this.panoProgram, "u_invViewProj");
        this.p_panoCube = gl.getUniformLocation(this.panoProgram, "u_panoCube");
        this.p_opacity = gl.getUniformLocation(this.panoProgram, "u_opacity");

        // Reuse a fullscreen quad
        this.panoVAO = gl.createVertexArray();
        gl.bindVertexArray(this.panoVAO);
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
            -1, -1,  1, -1,  -1, 1,  -1, 1,  1, -1,  1, 1,
        ]), gl.STATIC_DRAW);
        const loc = gl.getAttribLocation(this.panoProgram, "a_pos");
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
        gl.bindVertexArray(null);
    }

    _initPickBuffer() {
        // Mouse position for hover detection
        this._mouseX = 0;
        this._mouseY = 0;

        this.canvas.addEventListener("mousemove", (e) => {
            this._mouseX = e.clientX;
            this._mouseY = e.clientY;
        });
    }

    // Convert equirectangular image to cubemap (reused from SkyRenderer pattern)
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

    // Load a panorama image and convert to cubemap
    async loadPanorama(waypointId) {
        const wp = this.waypoints.find(w => w.id === waypointId);
        if (!wp) return null;

        // Return cached
        if (this.panoCubemaps[waypointId]) return this.panoCubemaps[waypointId];

        this.loadingPano = waypointId;

        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                const cubemap = this._equirectToCubemap(img, 1024);
                this.panoCubemaps[waypointId] = cubemap;
                this.loadingPano = null;
                console.log(`Panorama ${wp.file} loaded and converted to cubemap`);
                resolve(cubemap);
            };
            img.onerror = () => {
                console.error(`Failed to load panorama: ${wp.file}`);
                this.loadingPano = null;
                resolve(null);
            };
            img.src = `assets/panos/${wp.file}`;
        });
    }

    // Check if a waypoint marker is under the mouse (screen-space distance)
    hitTest(projMatrix, viewMatrix, mouseX, mouseY) {
        const w = this.canvas.width;
        const h = this.canvas.height;

        // Mouse in NDC
        const ndcX = (mouseX / this.canvas.clientWidth) * 2 - 1;
        const ndcY = 1 - (mouseY / this.canvas.clientHeight) * 2;

        let closest = null;
        let closestDist = 30; // max pixel distance for hit

        for (const wp of this.waypoints) {
            // Project waypoint to screen
            const p = wp.pos;
            // view * pos
            const vx = viewMatrix[0]*p[0] + viewMatrix[4]*p[1] + viewMatrix[8]*p[2] + viewMatrix[12];
            const vy = viewMatrix[1]*p[0] + viewMatrix[5]*p[1] + viewMatrix[9]*p[2] + viewMatrix[13];
            const vz = viewMatrix[2]*p[0] + viewMatrix[6]*p[1] + viewMatrix[10]*p[2] + viewMatrix[14];
            const vw = viewMatrix[3]*p[0] + viewMatrix[7]*p[1] + viewMatrix[11]*p[2] + viewMatrix[15];

            if (vw < 0.1) continue; // behind camera

            // proj * view_pos
            const cx = projMatrix[0]*vx + projMatrix[4]*vy + projMatrix[8]*vz + projMatrix[12]*vw;
            const cy = projMatrix[1]*vx + projMatrix[5]*vy + projMatrix[9]*vz + projMatrix[13]*vw;
            const cw = projMatrix[3]*vx + projMatrix[7]*vy + projMatrix[11]*vz + projMatrix[15]*vw;

            if (cw < 0.1) continue;

            const sx = cx / cw;
            const sy = cy / cw;

            // Distance in pixels
            const px = (sx - ndcX) * w / 2;
            const py = (sy - ndcY) * h / 2;
            const dist = Math.sqrt(px * px + py * py);

            if (dist < closestDist) {
                closestDist = dist;
                closest = wp;
            }
        }

        return closest;
    }

    // Enter panorama mode for a waypoint
    async enterPanorama(waypointId, camera) {
        const wp = this.waypoints.find(w => w.id === waypointId);
        if (!wp) return;

        // Save camera state for restoration on exit
        this._savedCamera = {
            theta: camera.theta,
            phi: camera.phi,
            distance: camera.distance,
            target: [...camera.target],
        };

        this.currentWaypoint = wp;
        this.transitioning = true;
        this.transitionStart = performance.now();

        // Start loading panorama immediately
        const cubemapPromise = this.loadPanorama(waypointId);

        // Fly camera toward waypoint
        this._flyTarget = wp.pos;
        this._flyStartTarget = [...camera.target];
        this._flyStartDist = camera.distance;
        this._flyCamera = camera;

        // Wait for panorama to load
        await cubemapPromise;

        // Initialize panorama camera: keep horizontal direction but look at horizon
        this.panoTheta = camera.theta;
        this.panoPhi = 0; // horizon level, not inherited from orbit camera
        this.panoFov = 75;
    }

    exitPanorama(camera) {
        if (!this.active && !this.transitioning) return;
        this.active = false;
        this.transitioning = true;
        this.transitionStart = performance.now();
        this._exitingPano = true;
        this._exitCamera = camera;
        // Capture current camera state as starting point for restoration
        this._exitStartState = {
            theta: camera.theta,
            phi: camera.phi,
            distance: camera.distance,
            target: [...camera.target],
        };
    }

    // Update transition state. Returns panoOpacity (0 = splats only, 1 = pano only)
    update(camera) {
        if (!this.transitioning) return;

        const elapsed = performance.now() - this.transitionStart;

        if (this._exitingPano) {
            // Fade out panorama (faster)
            const fadeT = Math.min(1, elapsed / this.exitFadeDuration);
            // Ease out quad for smooth fade
            this.panoOpacity = 1 - fadeT * fadeT;

            // Smoothly restore camera position (slower, continues after fade)
            const camT = Math.min(1, elapsed / this.exitCameraRestoreDuration);
            // Ease in-out cubic for buttery smooth camera movement
            const easedCamT = camT < 0.5
                ? 4 * camT * camT * camT
                : 1 - Math.pow(-2 * camT + 2, 3) / 2;

            if (this._savedCamera && this._exitCamera) {
                const cam = this._exitCamera;
                const s = this._savedCamera;
                cam.theta = this._exitStartState.theta + (s.theta - this._exitStartState.theta) * easedCamT;
                cam.phi = this._exitStartState.phi + (s.phi - this._exitStartState.phi) * easedCamT;
                cam.distance = this._exitStartState.distance + (s.distance - this._exitStartState.distance) * easedCamT;
                cam.target[0] = this._exitStartState.target[0] + (s.target[0] - this._exitStartState.target[0]) * easedCamT;
                cam.target[1] = this._exitStartState.target[1] + (s.target[1] - this._exitStartState.target[1]) * easedCamT;
                cam.target[2] = this._exitStartState.target[2] + (s.target[2] - this._exitStartState.target[2]) * easedCamT;
                cam._dirty = true;
            }

            if (camT >= 1) {
                this.transitioning = false;
                this._exitingPano = false;
                this.panoOpacity = 0;
                this.currentWaypoint = null;
                this._savedCamera = null;
                this._exitCamera = null;
                this._exitStartState = null;
            }
            return;
        }

        // Entering: fly camera, then fade in panorama
        if (elapsed < this.transitionDuration) {
            // Flying phase
            let t = elapsed / this.transitionDuration;
            t = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

            // Move camera target toward waypoint
            const wp = this._flyTarget;
            const st = this._flyStartTarget;
            camera.target[0] = st[0] + (wp[0] - st[0]) * t;
            camera.target[1] = st[1] + (wp[1] - st[1]) * t;
            camera.target[2] = st[2] + (wp[2] - st[2]) * t;

            // Zoom in
            camera.distance = this._flyStartDist * (1 - t * 0.7);
            camera._dirty = true;
        } else {
            // Crossfade phase
            const fadeElapsed = elapsed - this.transitionDuration;
            const t = Math.min(1, fadeElapsed / this.fadeDuration);
            this.panoOpacity = t;

            if (t >= 1) {
                this.active = true;
                this.transitioning = false;
            }
        }
    }

    // Render waypoint markers (call during splat rendering phase)
    renderMarkers(projMatrix, viewMatrix, time) {
        if (this.active) {
            this._label.style.display = "none";
            return; // hide markers in pano mode
        }

        const gl = this.gl;

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.disable(gl.DEPTH_TEST);

        gl.useProgram(this.markerProgram);
        gl.uniformMatrix4fv(this.m_projection, false, projMatrix);
        gl.uniformMatrix4fv(this.m_view, false, viewMatrix);
        gl.uniform2f(this.m_viewport, this.canvas.width, this.canvas.height);

        const pulse = Math.sin(time * 3) * 0.5 + 0.5;

        // Check hover
        const hovered = this.hitTest(projMatrix, viewMatrix, this._mouseX, this._mouseY);
        this.hoveredWaypoint = hovered;
        this.canvas.style.cursor = hovered ? "pointer" : "";

        // Show/hide hover label
        if (hovered) {
            // Project hovered waypoint to screen for label position
            const screenPos = this._projectToScreen(hovered.pos, projMatrix, viewMatrix);
            if (screenPos) {
                this._label.style.display = "block";
                this._label.textContent = `Scan ${hovered.id}`;
                this._label.style.left = `${screenPos[0] + 15}px`;
                this._label.style.top = `${screenPos[1] - 12}px`;
            }
        } else {
            this._label.style.display = "none";
        }

        gl.bindVertexArray(this.markerVAO);

        for (const wp of this.waypoints) {
            gl.uniform3f(this.m_markerPos, wp.pos[0], wp.pos[1], wp.pos[2]);
            gl.uniform1f(this.m_pulse, pulse);
            gl.uniform1f(this.m_hover, hovered && hovered.id === wp.id ? 1.0 : 0.0);
            gl.uniform1f(this.m_scale, 1.0);
            gl.drawArrays(gl.TRIANGLES, 0, 6);
        }

        gl.bindVertexArray(null);
    }

    // Project a 3D point to screen coordinates
    _projectToScreen(pos, projMatrix, viewMatrix) {
        const p = pos;
        const vx = viewMatrix[0]*p[0] + viewMatrix[4]*p[1] + viewMatrix[8]*p[2] + viewMatrix[12];
        const vy = viewMatrix[1]*p[0] + viewMatrix[5]*p[1] + viewMatrix[9]*p[2] + viewMatrix[13];
        const vz = viewMatrix[2]*p[0] + viewMatrix[6]*p[1] + viewMatrix[10]*p[2] + viewMatrix[14];
        const vw = viewMatrix[3]*p[0] + viewMatrix[7]*p[1] + viewMatrix[11]*p[2] + viewMatrix[15];
        if (vw < 0.1) return null;

        const cx = projMatrix[0]*vx + projMatrix[4]*vy + projMatrix[8]*vz + projMatrix[12]*vw;
        const cy = projMatrix[1]*vx + projMatrix[5]*vy + projMatrix[9]*vz + projMatrix[13]*vw;
        const cw = projMatrix[3]*vx + projMatrix[7]*vy + projMatrix[11]*vz + projMatrix[15]*vw;
        if (cw < 0.1) return null;

        const ndcX = cx / cw;
        const ndcY = cy / cw;

        const screenX = (ndcX * 0.5 + 0.5) * this.canvas.clientWidth;
        const screenY = (1 - (ndcY * 0.5 + 0.5)) * this.canvas.clientHeight;
        return [screenX, screenY];
    }

    // Render panorama overlay (call after splats)
    renderPanorama(mainProjMatrix, mainViewMatrix) {
        if (this.panoOpacity <= 0) return;
        if (!this.currentWaypoint) return;

        const cubemap = this.panoCubemaps[this.currentWaypoint.id];
        if (!cubemap) return;

        const gl = this.gl;

        // Build panorama view-projection (first-person, rotation only)
        const aspect = this.canvas.width / this.canvas.height;
        const fovRad = this.panoFov * Math.PI / 180;
        const f = 1 / Math.tan(fovRad / 2);
        const near = 0.1, far = 100.0;
        const nf = 1 / (near - far);
        const proj = new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) * nf, -1,
            0, 0, 2 * far * near * nf, 0,
        ]);

        // Rotation-only view matrix from pano angles
        const ct = Math.cos(this.panoTheta), st = Math.sin(this.panoTheta);
        const cp = Math.cos(this.panoPhi), sp = Math.sin(this.panoPhi);

        const eye = [cp * st, sp, cp * ct];
        const target = [0, 0, 0];
        const up = [0, 1, 0];

        // Simple lookAt for rotation only
        const zx = eye[0], zy = eye[1], zz = eye[2];
        let len = Math.sqrt(zx*zx + zy*zy + zz*zz);
        const z = [zx/len, zy/len, zz/len];
        const xx = up[1]*z[2] - up[2]*z[1];
        const xy = up[2]*z[0] - up[0]*z[2];
        const xz = up[0]*z[1] - up[1]*z[0];
        len = Math.sqrt(xx*xx + xy*xy + xz*xz);
        const x = [xx/len, xy/len, xz/len];
        const y = [z[1]*x[2]-z[2]*x[1], z[2]*x[0]-z[0]*x[2], z[0]*x[1]-z[1]*x[0]];

        const view = new Float32Array([
            x[0], y[0], z[0], 0,
            x[1], y[1], z[1], 0,
            x[2], y[2], z[2], 0,
            0, 0, 0, 1,
        ]);

        // VP = P * V (using mat4Multiply convention: mat4Multiply(a,b) = b*a)
        const vp = new Float32Array(16);
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                vp[i*4+j] = 0;
                for (let k = 0; k < 4; k++) {
                    vp[i*4+j] += proj[k*4+j] * view[i*4+k];
                }
            }
        }

        // Invert VP for ray reconstruction
        const invVP = mat4InvertLocal(vp);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.disable(gl.DEPTH_TEST);

        gl.useProgram(this.panoProgram);
        gl.uniformMatrix4fv(this.p_invViewProj, false, invVP);
        gl.uniform1f(this.p_opacity, this.panoOpacity);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_CUBE_MAP, cubemap);
        gl.uniform1i(this.p_panoCube, 0);

        gl.bindVertexArray(this.panoVAO);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        gl.bindVertexArray(null);
    }

    // Handle panorama look-around controls
    handlePanoInput(dx, dy, scrollDelta) {
        if (!this.active) return;
        // Street View style: drag to "grab and pull" the scene
        this.panoTheta -= dx * 0.003;
        this.panoPhi = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01,
            this.panoPhi + dy * 0.003));

        if (scrollDelta) {
            this.panoFov = Math.max(30, Math.min(110, this.panoFov + scrollDelta * 0.05));
        }
    }
}

// Local mat4 invert (so module is self-contained)
function mat4InvertLocal(m) {
    const inv = new Float32Array(16);
    const n11=m[0],n21=m[1],n31=m[2],n41=m[3];
    const n12=m[4],n22=m[5],n32=m[6],n42=m[7];
    const n13=m[8],n23=m[9],n33=m[10],n43=m[11];
    const n14=m[12],n24=m[13],n34=m[14],n44=m[15];
    const t11=n23*n34*n42-n24*n33*n42+n24*n32*n43-n22*n34*n43-n23*n32*n44+n22*n33*n44;
    const t12=n14*n33*n42-n13*n34*n42-n14*n32*n43+n12*n34*n43+n13*n32*n44-n12*n33*n44;
    const t13=n13*n24*n42-n14*n23*n42+n14*n22*n43-n12*n24*n43-n13*n22*n44+n12*n23*n44;
    const t14=n14*n23*n32-n13*n24*n32-n14*n22*n33+n12*n24*n33+n13*n22*n34-n12*n23*n34;
    const det=n11*t11+n21*t12+n31*t13+n41*t14;
    if(det===0)return new Float32Array(16);
    const d=1/det;
    inv[0]=t11*d;inv[1]=(n24*n33*n41-n23*n34*n41-n24*n31*n43+n21*n34*n43+n23*n31*n44-n21*n33*n44)*d;
    inv[2]=(n22*n34*n41-n24*n32*n41+n24*n31*n42-n21*n34*n42-n22*n31*n44+n21*n32*n44)*d;
    inv[3]=(n23*n32*n41-n22*n33*n41-n23*n31*n42+n21*n33*n42+n22*n31*n43-n21*n32*n43)*d;
    inv[4]=t12*d;inv[5]=(n13*n34*n41-n14*n33*n41+n14*n31*n43-n11*n34*n43-n13*n31*n44+n11*n33*n44)*d;
    inv[6]=(n14*n32*n41-n12*n34*n41-n14*n31*n42+n11*n34*n42+n12*n31*n44-n11*n32*n44)*d;
    inv[7]=(n12*n33*n41-n13*n32*n41+n13*n31*n42-n11*n33*n42-n12*n31*n43+n11*n32*n43)*d;
    inv[8]=t13*d;inv[9]=(n14*n23*n41-n13*n24*n41-n14*n21*n43+n11*n24*n43+n13*n21*n44-n11*n23*n44)*d;
    inv[10]=(n12*n24*n41-n14*n22*n41+n14*n21*n42-n11*n24*n42-n12*n21*n44+n11*n22*n44)*d;
    inv[11]=(n13*n22*n41-n12*n23*n41-n13*n21*n42+n11*n23*n42+n12*n21*n43-n11*n22*n43)*d;
    inv[12]=t14*d;inv[13]=(n13*n24*n31-n14*n23*n31+n14*n21*n33-n11*n24*n33-n13*n21*n34+n11*n23*n34)*d;
    inv[14]=(n14*n22*n31-n12*n24*n31-n14*n21*n32+n11*n24*n32+n12*n21*n34-n11*n22*n34)*d;
    inv[15]=(n12*n23*n31-n13*n22*n31+n13*n21*n32-n11*n23*n32-n12*n21*n33+n11*n22*n33)*d;
    return inv;
}
