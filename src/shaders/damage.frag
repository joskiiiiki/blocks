#version 330

// --- frag ---
uniform float u_intensity; // 0.0 (no damage) → 1.0 (just hit)
uniform float u_time; // game time in seconds, for blood drip animation
uniform vec2 u_resolution; // viewport size in pixels

in vec2 uv;
out vec4 fragColor;

// --- noise helpers ---
float hash(vec2 p) {
    p = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + 17.5);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f); // smoothstep
    return mix(
        mix(hash(i), hash(i + vec2(1, 0)), f.x),
        mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x),
        f.y
    );
}

// --- vignette ---
float vignette(vec2 uv, float strength) {
    vec2 d = uv - 0.5;
    d.x *= u_resolution.x / u_resolution.y; // correct aspect
    return 1.0 - dot(d, d) * strength;
}

// --- blood drip ---
// Returns blood alpha at this UV given drip params
float blood_drip(vec2 uv, float seed, float speed, float width, float start_y) {
    float drip_x = fract(seed); // fixed x position
    float dx = abs(uv.x - drip_x);
    if (dx > width) return 0.0;

    float drip_len = 0.15 + 0.1 * fract(seed * 7.3);
    float tip_y = start_y + fract(seed * 3.7) * 0.1 + u_time * speed * 0.04;
    tip_y = min(tip_y, 0.95); // clamp to screen

    float dy = uv.y - (1.0 - tip_y); // drips downward from top
    if (dy < -drip_len || dy > 0.0) return 0.0;

    float along = 1.0 - (-dy / drip_len); // 0 at tip, 1 at root
    float shape = (1.0 - dx / width); // taper width
    shape *= shape;
    float bulge = sin(along * 3.14159) * 0.5 + 0.5; // fatter in middle
    return shape * bulge * along;
}

// --- splatter blob ---
float splatter(vec2 uv, vec2 center, float radius, float seed) {
    vec2 d = uv - center;
    float r = length(d);
    // perturb edge with noise for organic shape
    float angle = atan(d.y, d.x);
    float perturb = noise(vec2(angle * 2.0 + seed, seed)) * 0.4 + 0.8;
    return smoothstep(radius * perturb, radius * perturb * 0.4, r);
}

void main() {
    if (u_intensity <= 0.0) {
        fragColor = vec4(0.0);
        return;
    }

    float intensity = clamp(u_intensity, 0.0, 1.0);

    // --- vignette ring ---
    float vig = vignette(uv, 4.5 * intensity);
    vig = clamp(1.0 - vig, 0.0, 1.0);
    vig = pow(vig, 1.5);

    // inner glow pulse
    float pulse = sin(u_time * 6.0) * 0.04 * intensity;
    vig += pulse;

    // --- edge noise for organic border ---
    float edge_noise = noise(uv * 6.0 + u_time * 0.3) * 0.15 * intensity;
    vig += edge_noise * vig; // only adds to already-vignetted areas

    // --- blood drips from top edge ---
    float blood = 0.0;

    // only show drips when intensity is high enough
    if (intensity > 0.3) {
        float drip_intensity = smoothstep(0.3, 0.7, intensity);

        blood += blood_drip(uv, 0.12, 0.8, 0.012, 0.02) * drip_intensity;
        blood += blood_drip(uv, 0.31, 1.1, 0.009, 0.05) * drip_intensity;
        blood += blood_drip(uv, 0.58, 0.6, 0.014, 0.01) * drip_intensity;
        blood += blood_drip(uv, 0.74, 0.9, 0.010, 0.03) * drip_intensity;
        blood += blood_drip(uv, 0.89, 1.3, 0.008, 0.06) * drip_intensity;
        blood += blood_drip(uv, 0.43, 0.7, 0.011, 0.04) * drip_intensity;

        // splatter blobs near top corners
        blood += splatter(uv, vec2(0.05, 0.04), 0.04, 1.0) * drip_intensity * 0.8;
        blood += splatter(uv, vec2(0.92, 0.06), 0.035, 2.0) * drip_intensity * 0.8;
        blood += splatter(uv, vec2(0.15, 0.09), 0.02, 3.0) * drip_intensity * 0.6;
        blood += splatter(uv, vec2(0.78, 0.03), 0.025, 4.0) * drip_intensity * 0.7;
    }

    blood = clamp(blood, 0.0, 1.0);

    // --- compose ---
    vec3 blood_color = vec3(0.55, 0.0, 0.02);
    vec3 vignette_color = vec3(0.7, 0.0, 0.05);

    vec3 color = mix(vignette_color, blood_color, blood);
    float alpha = clamp(vig * 0.75 + blood * 0.95, 0.0, 0.92);
    alpha *= intensity;

    fragColor = vec4(color, alpha);
}
