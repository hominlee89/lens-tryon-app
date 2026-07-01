class SoftFocusEngine {
    constructor(videoElement, canvasElement) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.gl = this.canvas.getContext('webgl', { preserveDrawingBuffer: true });

        if (!this.gl) {
            console.error("WebGL not supported");
            return;
        }

        this.initShaders();
        this.initBuffers();
        this.initTextures();
    }

    initShaders() {
        const vsSource = `
            attribute vec4 aPosition;
            attribute vec2 aTexCoord;
            varying vec2 vTexCoord;
            uniform mat3 uTexMatrix;

            void main() {
                gl_Position = aPosition;
                vTexCoord = (uTexMatrix * vec3(aTexCoord, 1.0)).xy;
            }
        `;

        const fsSource = `
            precision mediump float;

            varying vec2 vTexCoord;
            uniform sampler2D uSampler;
            uniform sampler2D uFaceMaskSampler;
            uniform vec2 uResolution;
            uniform vec2 uMaskTexel;
            uniform float uSoftFocusStrength;
            uniform int uFaceCount;

            uniform vec2 uForehead0;
            uniform vec2 uLeftCheek0;
            uniform vec2 uRightCheek0;
            uniform vec2 uChin0;
            uniform vec2 uFaceCenter0;
            uniform vec2 uLeftEyeCenter0;
            uniform vec2 uRightEyeCenter0;
            uniform vec2 uLeftJaw0;
            uniform vec2 uRightJaw0;
            uniform vec2 uMouthCenter0;

            uniform vec2 uForehead1;
            uniform vec2 uLeftCheek1;
            uniform vec2 uRightCheek1;
            uniform vec2 uChin1;
            uniform vec2 uFaceCenter1;
            uniform vec2 uLeftEyeCenter1;
            uniform vec2 uRightEyeCenter1;
            uniform vec2 uLeftJaw1;
            uniform vec2 uRightJaw1;
            uniform vec2 uMouthCenter1;

            float luma(vec3 rgb) {
                return dot(rgb, vec3(0.299, 0.587, 0.114));
            }

            float skinScore(vec3 rgb) {
                float r = rgb.r;
                float g = rgb.g;
                float b = rgb.b;
                float y = luma(rgb);
                float cb = b - y;
                float cr = r - y;

                float lightGate = smoothstep(0.045, 0.15, y) * (1.0 - smoothstep(0.94, 1.0, y));
                float warmBalance = smoothstep(-0.11, 0.15, r - g) * smoothstep(-0.19, 0.19, g - b);
                float redBlueBalance = smoothstep(-0.09, 0.19, r - b);
                float chromaSkin = smoothstep(-0.05, 0.17, cr) * (1.0 - smoothstep(0.20, 0.40, abs(cb + 0.03)));

                return clamp(max(warmBalance * redBlueBalance, chromaSkin) * lightGate, 0.0, 1.0);
            }

            vec3 matchLuma(vec3 reference, vec3 candidate, float amount) {
                return clamp(candidate + vec3((luma(reference) - luma(candidate)) * amount), 0.0, 1.0);
            }

            vec3 calmRedness(vec3 color, float redMask) {
                vec3 calmer = color;
                float redExcess = max(color.r - max(color.g, color.b), 0.0);
                calmer.r = color.r - redExcess * clamp(redMask * 0.18, 0.0, 0.18);
                return matchLuma(color, calmer, 0.55);
            }

            float meshMaskAt(vec2 uv) {
                vec2 px = uMaskTexel;
                float mask = texture2D(uFaceMaskSampler, uv).r * 0.28;
                mask += texture2D(uFaceMaskSampler, uv + px * vec2(2.0, 0.0)).r * 0.10;
                mask += texture2D(uFaceMaskSampler, uv + px * vec2(-2.0, 0.0)).r * 0.10;
                mask += texture2D(uFaceMaskSampler, uv + px * vec2(0.0, 2.0)).r * 0.10;
                mask += texture2D(uFaceMaskSampler, uv + px * vec2(0.0, -2.0)).r * 0.10;
                mask += texture2D(uFaceMaskSampler, uv + px * vec2(4.0, 4.0)).r * 0.055;
                mask += texture2D(uFaceMaskSampler, uv + px * vec2(-4.0, 4.0)).r * 0.055;
                mask += texture2D(uFaceMaskSampler, uv + px * vec2(4.0, -4.0)).r * 0.055;
                mask += texture2D(uFaceMaskSampler, uv + px * vec2(-4.0, -4.0)).r * 0.055;
                mask += texture2D(uFaceMaskSampler, uv + px * vec2(8.0, 0.0)).r * 0.035;
                mask += texture2D(uFaceMaskSampler, uv + px * vec2(-8.0, 0.0)).r * 0.035;
                mask += texture2D(uFaceMaskSampler, uv + px * vec2(0.0, 8.0)).r * 0.035;
                mask += texture2D(uFaceMaskSampler, uv + px * vec2(0.0, -8.0)).r * 0.035;

                return smoothstep(0.28, 0.92, clamp(mask, 0.0, 1.0));
            }

            float faceMaskAt(
                vec2 uv,
                vec2 forehead,
                vec2 leftCheek,
                vec2 rightCheek,
                vec2 chin,
                vec2 faceCenter,
                vec2 leftJaw,
                vec2 rightJaw
            ) {
                float faceWidth = max(distance(leftJaw, rightJaw), 0.10);
                float faceHeight = max(distance(forehead, chin), 0.14);
                vec2 center = mix(faceCenter, chin, 0.22);
                vec2 p = uv - center;

                float ellipse = (p.x * p.x) / (faceWidth * faceWidth * 0.31) +
                    (p.y * p.y) / (faceHeight * faceHeight * 0.28);
                float base = 1.0 - smoothstep(0.70, 1.02, ellipse);

                float cheekMask = (1.0 - smoothstep(0.055, 0.17, distance(uv, leftCheek))) +
                    (1.0 - smoothstep(0.055, 0.17, distance(uv, rightCheek)));
                float foreheadMask = 1.0 - smoothstep(0.045, 0.13, distance(uv, forehead));
                float chinMask = 1.0 - smoothstep(0.045, 0.13, distance(uv, chin));

                float verticalGuard = smoothstep(-0.18, 0.06, uv.y - forehead.y) *
                    (1.0 - smoothstep(0.07, 0.22, uv.y - chin.y));

                return clamp((base + cheekMask * 0.36 + foreheadMask * 0.20 + chinMask * 0.16) * verticalGuard, 0.0, 1.0);
            }

            vec2 faceSlimWarpAt(
                vec2 uv,
                vec2 forehead,
                vec2 chin,
                vec2 faceCenter,
                vec2 leftJaw,
                vec2 rightJaw
            ) {
                vec2 faceMid = mix(faceCenter, chin, 0.30);
                float faceWidth = max(distance(leftJaw, rightJaw), 0.10);
                float faceHeight = max(distance(forehead, chin), 0.14);
                vec2 p = uv - faceMid;

                float vertical = 1.0 - smoothstep(0.08, 0.62, abs(p.y) / faceHeight);
                float lowerFace = smoothstep(-0.05, 0.36, (uv.y - faceCenter.y) / faceHeight);
                float innerZone = 1.0 - smoothstep(0.24, 0.58, abs(p.x) / faceWidth);
                float contourZone = vertical * lowerFace * (1.0 - innerZone);
                float side = sign(p.x);

                uv.x += side * contourZone * faceWidth * 0.038 * uSoftFocusStrength;
                uv.y -= contourZone * max(0.0, p.y) * 0.012 * uSoftFocusStrength;

                return clamp(uv, vec2(0.001), vec2(0.999));
            }

            vec2 featureMagnify(vec2 uv, vec2 center, vec2 radius, float amount) {
                vec2 delta = uv - center;
                vec2 normalized = delta / radius;
                float dist = length(normalized);
                float falloff = pow(1.0 - smoothstep(0.0, 1.0, dist), 1.45);

                return center + delta * (1.0 - amount * falloff);
            }

            vec2 eyeWarpAt(vec2 uv, vec2 leftEyeCenter, vec2 rightEyeCenter, vec2 leftJaw, vec2 rightJaw) {
                float faceWidth = max(distance(leftJaw, rightJaw), 0.10);
                vec2 eyeRadius = vec2(faceWidth * 0.165, faceWidth * 0.098);
                float amount = 0.090 * uSoftFocusStrength;

                uv = featureMagnify(uv, leftEyeCenter, eyeRadius, amount);
                uv = featureMagnify(uv, rightEyeCenter, eyeRadius, amount);

                return clamp(uv, vec2(0.001), vec2(0.999));
            }

            vec3 edgeAwareSmooth(sampler2D sampler, vec2 uv, vec2 res, float sigmaS, float sigmaI) {
                const float radius = 3.0;
                vec3 centerColor = texture2D(sampler, uv).rgb;
                vec3 sumColor = vec3(0.0);
                float sumWeight = 0.0;

                for (float x = -radius; x <= radius; x += 1.0) {
                    for (float y = -radius; y <= radius; y += 1.0) {
                        vec2 offset = vec2(x, y) / res;
                        vec3 sampleColor = texture2D(sampler, uv + offset).rgb;
                        float weightS = exp(-dot(vec2(x, y), vec2(x, y)) / (2.0 * sigmaS * sigmaS));
                        float weightI = exp(-distance(centerColor, sampleColor) / (2.0 * sigmaI * sigmaI));
                        float weight = weightS * weightI;
                        sumColor += sampleColor * weight;
                        sumWeight += weight;
                    }
                }

                return sumColor / sumWeight;
            }

            vec3 broadToneSmooth(sampler2D sampler, vec2 uv, vec2 res) {
                vec2 px = 1.0 / res;
                vec3 c = texture2D(sampler, uv).rgb * 0.20;

                c += texture2D(sampler, uv + px * vec2(3.0, 0.0)).rgb * 0.09;
                c += texture2D(sampler, uv + px * vec2(-3.0, 0.0)).rgb * 0.09;
                c += texture2D(sampler, uv + px * vec2(0.0, 3.0)).rgb * 0.09;
                c += texture2D(sampler, uv + px * vec2(0.0, -3.0)).rgb * 0.09;
                c += texture2D(sampler, uv + px * vec2(4.0, 4.0)).rgb * 0.07;
                c += texture2D(sampler, uv + px * vec2(-4.0, 4.0)).rgb * 0.07;
                c += texture2D(sampler, uv + px * vec2(4.0, -4.0)).rgb * 0.07;
                c += texture2D(sampler, uv + px * vec2(-4.0, -4.0)).rgb * 0.07;
                c += texture2D(sampler, uv + px * vec2(7.0, 0.0)).rgb * 0.04;
                c += texture2D(sampler, uv + px * vec2(-7.0, 0.0)).rgb * 0.04;
                c += texture2D(sampler, uv + px * vec2(0.0, 7.0)).rgb * 0.04;
                c += texture2D(sampler, uv + px * vec2(0.0, -7.0)).rgb * 0.04;

                return c;
            }

            float eyeCoreGuardAt(vec2 uv, vec2 leftEyeCenter, vec2 rightEyeCenter) {
                float dLeft = distance(uv, leftEyeCenter);
                float dRight = distance(uv, rightEyeCenter);
                return min(smoothstep(0.030, 0.068, dLeft), smoothstep(0.030, 0.068, dRight));
            }

            float underEyeMaskAt(vec2 uv, vec3 rgb, vec2 leftEyeCenter, vec2 rightEyeCenter) {
                float skin = skinScore(rgb);
                float left = 1.0 - smoothstep(0.030, 0.145, distance(uv, leftEyeCenter));
                float right = 1.0 - smoothstep(0.030, 0.145, distance(uv, rightEyeCenter));
                float belowLeft = smoothstep(-0.018, 0.082, uv.y - leftEyeCenter.y);
                float belowRight = smoothstep(-0.018, 0.082, uv.y - rightEyeCenter.y);
                float awayFromEyeCore = min(
                    smoothstep(0.026, 0.054, distance(uv, leftEyeCenter)),
                    smoothstep(0.026, 0.054, distance(uv, rightEyeCenter))
                );

                return clamp((left * belowLeft + right * belowRight) * awayFromEyeCore * skin, 0.0, 1.0);
            }

            vec4 beautyForFace(
                vec2 baseUv,
                vec4 original,
                float faceMask,
                vec2 forehead,
                vec2 leftCheek,
                vec2 rightCheek,
                vec2 chin,
                vec2 faceCenter,
                vec2 leftEyeCenter,
                vec2 rightEyeCenter,
                vec2 leftJaw,
                vec2 rightJaw,
                vec2 mouthCenter
            ) {
                vec2 warpedUv = faceSlimWarpAt(baseUv, forehead, chin, faceCenter, leftJaw, rightJaw);
                warpedUv = eyeWarpAt(warpedUv, leftEyeCenter, rightEyeCenter, leftJaw, rightJaw);

                float sourceMeshMask = meshMaskAt(warpedUv);
                float confinedMask = min(faceMask, sourceMeshMask);
                float effectMask = smoothstep(0.48, 0.92, confinedMask);
                vec4 warped = texture2D(uSampler, warpedUv);

                float shapeWeight = effectMask * uSoftFocusStrength * 0.46;
                vec3 finalColor = mix(original.rgb, warped.rgb, shapeWeight);

                float skin = max(skinScore(warped.rgb), effectMask * 0.18);
                float eyeGuard = eyeCoreGuardAt(warpedUv, leftEyeCenter, rightEyeCenter);
                float mouthGuard = smoothstep(0.038, 0.105, distance(warpedUv, mouthCenter));
                float skinMask = effectMask * skin * eyeGuard * mouthGuard;

                if (skinMask > 0.01) {
                    vec3 smoothed = edgeAwareSmooth(uSampler, warpedUv, uResolution, 4.0, 0.28);
                    vec3 broad = broadToneSmooth(uSampler, warpedUv, uResolution);
                    float detail = distance(warped.rgb, smoothed);
                    float redSpot = smoothstep(0.024, 0.130, warped.r - max(warped.g, warped.b));
                    float darkCrease = smoothstep(0.016, 0.092, luma(broad) - luma(warped.rgb));
                    float poreNoise = smoothstep(0.010, 0.072, detail);
                    float blemishMask = clamp(skinMask * poreNoise * (0.58 + redSpot * 1.60 + darkCrease * 1.05), 0.0, 1.0);
                    float toneMask = clamp(skinMask * (0.12 + blemishMask * 0.42 + redSpot * 0.16), 0.0, 1.0);

                    vec3 broadMatched = matchLuma(warped.rgb, broad, 0.96);
                    vec3 smoothMatched = matchLuma(warped.rgb, smoothed, 0.94);
                    vec3 evenTone = mix(smoothMatched, broadMatched, 0.34);
                    evenTone = calmRedness(evenTone, redSpot);
                    evenTone = matchLuma(warped.rgb, evenTone, 0.96);

                    float eyeMask = underEyeMaskAt(warpedUv, warped.rgb, leftEyeCenter, rightEyeCenter) * effectMask;
                    float eyeCreaseMask = clamp(eyeMask * (0.54 + darkCrease * 1.70 + poreNoise * 0.26), 0.0, 1.0);
                    vec3 eyeSmooth = matchLuma(warped.rgb, mix(smoothMatched, broadMatched, 0.48), 0.96);

                    vec3 corrected = warped.rgb;
                    corrected = mix(corrected, smoothMatched, clamp(skinMask * uSoftFocusStrength * 0.52, 0.0, 1.0));
                    corrected = mix(corrected, evenTone, clamp(toneMask * uSoftFocusStrength * 0.34, 0.0, 1.0));
                    corrected = mix(corrected, evenTone, clamp(blemishMask * uSoftFocusStrength * 0.86, 0.0, 1.0));
                    corrected = mix(corrected, eyeSmooth, clamp(eyeCreaseMask * uSoftFocusStrength * 0.82, 0.0, 1.0));
                    corrected = calmRedness(corrected, redSpot * skinMask);
                    corrected = matchLuma(warped.rgb, corrected, 0.98);
                    corrected = clamp(corrected, 0.0, 1.0);

                    finalColor = mix(finalColor, corrected, clamp(skinMask * 0.56 + blemishMask * 0.44 + eyeCreaseMask * 0.58, 0.0, 1.0));
                }

                return vec4(finalColor, original.a);
            }

            void main() {
                vec2 baseUv = vTexCoord;
                vec4 original = texture2D(uSampler, baseUv);

                if (uSoftFocusStrength <= 0.001 || uFaceCount <= 0) {
                    gl_FragColor = original;
                    return;
                }

                float mask0 = 0.0;
                float mask1 = 0.0;
                float meshMask = meshMaskAt(baseUv);

                if (uFaceCount >= 1) {
                    mask0 = faceMaskAt(baseUv, uForehead0, uLeftCheek0, uRightCheek0, uChin0, uFaceCenter0, uLeftJaw0, uRightJaw0);
                }

                if (uFaceCount >= 2) {
                    mask1 = faceMaskAt(baseUv, uForehead1, uLeftCheek1, uRightCheek1, uChin1, uFaceCenter1, uLeftJaw1, uRightJaw1);
                }

                if (meshMask <= 0.015) {
                    gl_FragColor = original;
                    return;
                }

                if (mask1 > mask0) {
                    gl_FragColor = beautyForFace(
                        baseUv,
                        original,
                        meshMask,
                        uForehead1,
                        uLeftCheek1,
                        uRightCheek1,
                        uChin1,
                        uFaceCenter1,
                        uLeftEyeCenter1,
                        uRightEyeCenter1,
                        uLeftJaw1,
                        uRightJaw1,
                        uMouthCenter1
                    );
                } else {
                    gl_FragColor = beautyForFace(
                        baseUv,
                        original,
                        meshMask,
                        uForehead0,
                        uLeftCheek0,
                        uRightCheek0,
                        uChin0,
                        uFaceCenter0,
                        uLeftEyeCenter0,
                        uRightEyeCenter0,
                        uLeftJaw0,
                        uRightJaw0,
                        uMouthCenter0
                    );
                }
            }
        `;

        this.program = this.createProgram(vsSource, fsSource);
    }

    createProgram(vsSource, fsSource) {
        const vs = this.compileShader(this.gl.VERTEX_SHADER, vsSource);
        const fs = this.compileShader(this.gl.FRAGMENT_SHADER, fsSource);
        const program = this.gl.createProgram();
        this.gl.attachShader(program, vs);
        this.gl.attachShader(program, fs);
        this.gl.linkProgram(program);

        if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
            console.error("WebGL program link error:", this.gl.getProgramInfoLog(program));
        }

        return program;
    }

    compileShader(type, source) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);

        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            console.error("WebGL shader compile error:", this.gl.getShaderInfoLog(shader));
        }

        return shader;
    }

    initBuffers() {
        const positions = new Float32Array([
            -1.0, 1.0,
            1.0, 1.0,
            -1.0, -1.0,
            1.0, -1.0,
        ]);
        const texCoords = new Float32Array([
            1.0, 0.0,
            0.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,
        ]);

        this.positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        this.texCoordBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.texCoordBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, texCoords, this.gl.STATIC_DRAW);
    }

    initTextures() {
        this.texture = this.gl.createTexture();
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.texture);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);

        this.faceMaskTexture = this.gl.createTexture();
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.faceMaskTexture);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
    }

    render(video, strength = 0.5, landmarksInput = [], faceMaskCanvas = null) {
        if (!this.gl) {
            const ctx = this.canvas.getContext('2d');
            if (ctx) {
                ctx.save();
                ctx.scale(-1, 1);
                ctx.drawImage(video, -this.canvas.width, 0, this.canvas.width, this.canvas.height);
                ctx.restore();
            }
            return;
        }

        const gl = this.gl;
        const faces = this.normalizeFaces(landmarksInput).slice(0, 2);

        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.useProgram(this.program);

        const aPosition = gl.getAttribLocation(this.program, "aPosition");
        gl.enableVertexAttribArray(aPosition);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
        gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 0, 0);

        const aTexCoord = gl.getAttribLocation(this.program, "aTexCoord");
        gl.enableVertexAttribArray(aTexCoord);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
        gl.vertexAttribPointer(aTexCoord, 2, gl.FLOAT, false, 0, 0);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);

        gl.uniform1i(gl.getUniformLocation(this.program, "uSampler"), 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.faceMaskTexture);
        if (faceMaskCanvas) {
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, faceMaskCanvas);
            gl.uniform2f(
                gl.getUniformLocation(this.program, "uMaskTexel"),
                1 / faceMaskCanvas.width,
                1 / faceMaskCanvas.height
            );
        } else {
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
            gl.uniform2f(gl.getUniformLocation(this.program, "uMaskTexel"), 1, 1);
        }
        gl.uniform1i(gl.getUniformLocation(this.program, "uFaceMaskSampler"), 1);

        gl.uniform2f(gl.getUniformLocation(this.program, "uResolution"), this.canvas.width, this.canvas.height);
        gl.uniform1f(gl.getUniformLocation(this.program, "uSoftFocusStrength"), strength);
        gl.uniform1i(gl.getUniformLocation(this.program, "uFaceCount"), faces.length);

        const transform = this.getTextureTransform(video);
        gl.uniformMatrix3fv(gl.getUniformLocation(this.program, "uTexMatrix"), false, transform.matrix);

        for (let index = 0; index < 2; index += 1) {
            this.setFaceUniforms(index, faces[index], transform);
        }

        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    normalizeFaces(landmarksInput) {
        if (!landmarksInput) return [];
        if (!Array.isArray(landmarksInput)) return [];
        if (!landmarksInput.length) return [];
        return Array.isArray(landmarksInput[0]) ? landmarksInput : [landmarksInput];
    }

    getTextureTransform(video) {
        const videoW = video.videoWidth || 1280;
        const videoH = video.videoHeight || 720;
        const canvasW = this.canvas.width;
        const canvasH = this.canvas.height;
        const videoAR = videoW / videoH;
        const canvasAR = canvasW / canvasH;

        let scaleX = 1;
        let scaleY = 1;
        let offsetX = 0;
        let offsetY = 0;

        if (canvasAR > videoAR) {
            scaleY = videoAR / canvasAR;
            offsetY = (1.0 - scaleY) / 2.0;
        } else {
            scaleX = canvasAR / videoAR;
            offsetX = (1.0 - scaleX) / 2.0;
        }

        return {
            scaleX,
            scaleY,
            offsetX,
            offsetY,
            matrix: new Float32Array([
                scaleX, 0, 0,
                0, scaleY, 0,
                offsetX, offsetY, 1
            ])
        };
    }

    setFaceUniforms(index, landmarks, transform) {
        const zero = { x: 0, y: 0 };

        if (!landmarks) {
            this.setFaceUniformGroup(index, {
                forehead: zero,
                leftCheek: zero,
                rightCheek: zero,
                chin: zero,
                faceCenter: zero,
                leftEyeCenter: zero,
                rightEyeCenter: zero,
                leftJaw: zero,
                rightJaw: zero,
                mouthCenter: zero
            });
            return;
        }

        const mapPoint = (point) => ({
            x: point.x * transform.scaleX + transform.offsetX,
            y: point.y * transform.scaleY + transform.offsetY
        });
        const averagePoint = (indices) => {
            const sum = indices.reduce((acc, landmarkIndex) => {
                acc.x += landmarks[landmarkIndex].x;
                acc.y += landmarks[landmarkIndex].y;
                return acc;
            }, { x: 0, y: 0 });

            return {
                x: sum.x / indices.length,
                y: sum.y / indices.length
            };
        };

        this.setFaceUniformGroup(index, {
            forehead: mapPoint(landmarks[151]),
            leftCheek: mapPoint(landmarks[423]),
            rightCheek: mapPoint(landmarks[203]),
            chin: mapPoint(landmarks[152]),
            faceCenter: mapPoint(landmarks[1]),
            leftEyeCenter: mapPoint(averagePoint([33, 133, 159, 145])),
            rightEyeCenter: mapPoint(averagePoint([263, 362, 386, 374])),
            leftJaw: mapPoint(landmarks[234]),
            rightJaw: mapPoint(landmarks[454]),
            mouthCenter: mapPoint(averagePoint([13, 14, 61, 291]))
        });
    }

    setFaceUniformGroup(index, points) {
        const suffix = String(index);

        this.setVec2(`uForehead${suffix}`, points.forehead);
        this.setVec2(`uLeftCheek${suffix}`, points.leftCheek);
        this.setVec2(`uRightCheek${suffix}`, points.rightCheek);
        this.setVec2(`uChin${suffix}`, points.chin);
        this.setVec2(`uFaceCenter${suffix}`, points.faceCenter);
        this.setVec2(`uLeftEyeCenter${suffix}`, points.leftEyeCenter);
        this.setVec2(`uRightEyeCenter${suffix}`, points.rightEyeCenter);
        this.setVec2(`uLeftJaw${suffix}`, points.leftJaw);
        this.setVec2(`uRightJaw${suffix}`, points.rightJaw);
        this.setVec2(`uMouthCenter${suffix}`, points.mouthCenter);
    }

    setVec2(name, point) {
        const location = this.gl.getUniformLocation(this.program, name);
        if (location) {
            this.gl.uniform2f(location, point.x, point.y);
        }
    }
}
