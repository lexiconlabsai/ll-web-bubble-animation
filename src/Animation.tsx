/* eslint-disable react/no-unknown-property */
import * as THREE from 'three'
import { button, useControls } from 'leva'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { useRef, useMemo, useState, useEffect, useCallback } from 'react'

type VoiceModeState = 'idle' | 'initializing' | 'listening' | 'thinking' | 'speaking'

const lerp = (start: number, end: number, t: number) => start * (1 - t) + end * t

const presetStates = {
  speaking: {
    innerRadius: 0.27,
    outerRadius: 0.34,
    sheetThickness: 0.03,
    noisiness: 7,
    timeScale: 2,
    overallScale: 2,
  },
  thinking: {
    innerRadius: 0.25,
    outerRadius: 0.41,
    sheetThickness: 0.03,
    noisiness: 6.5,
    timeScale: 5,
    overallScale: 2,
  },
  listening: {
    innerRadius: 0.25,
    outerRadius: 0.35,
    sheetThickness: 0.01,
    noisiness: 7,
    timeScale: 5,
    overallScale: 1.5,
  },
  idle: {
    innerRadius: 0.25,
    outerRadius: 0.28,
    sheetThickness: 0.025,
    noisiness: 9,
    timeScale: 5,
    overallScale: 1,
  },
  initializing: {
    innerRadius: 0.25,
    outerRadius: 0.28,
    sheetThickness: 0.025,
    noisiness: 9,
    timeScale: 9,
    overallScale: 1.3,
  },
}

const fragmentShader = `precision highp float;

uniform float INNER_RADIUS;
uniform float OUTER_RADIUS;
uniform float SHEET_THICKNESS;
uniform float NOISINESS;

vec4 INNER_COLOR = vec4(2., 0., 0., 1.);
vec4 OUTER_COLOR = vec4(0., 0., 2., 1.);

const int NUM_STEPS = 40;
uniform float TIME_SCALE;
varying vec2 vTexCoord;
uniform float uTime;

float trapezium(float x) {
  return min(1.0, max(0.0, 1.0 - abs(-mod(x, 1.0) * 3.0 + 1.0)) * 2.0);
}

vec3 colFromHue(float hue) {

  return vec3(trapezium(hue - 1.0 / 3.0), trapezium(hue), trapezium(hue + 1.0 / 3.0));
}
float cnoise3(float pos) {
  return (cos(pos / 2.0) * 0.2 + 1.0);
}

float cnoise2(float pos) {
  return (sin(pos * cnoise3(pos) / 2.0) * 0.2 + 1.0);
}

float cnoise(vec4 pos) {
  float x = pos.x * cnoise2(pos.y) + pos.w * 0.87123 + 82.52;
  float y = pos.y * cnoise2(pos.z) + pos.w * 0.78725 + 12.76;
  float z = pos.z * cnoise2(pos.x) + pos.w * 0.68201 + 42.03;
  return (sin(x) + sin(y) + sin(z)) / 3.0;
}

vec4 merge_colours(vec4 apply_this, vec4 on_top_of_this) {
  return on_top_of_this * (1.0 - apply_this.a) + apply_this * apply_this.a;
}

vec4 getdensity(vec3 pos) {
  float time = uTime * TIME_SCALE;

  vec3 samplePos = normalize(pos);
  vec4 inner_color = INNER_COLOR;
  vec4 outer_color = merge_colours(OUTER_COLOR, inner_color);

  float sample_ = (cnoise(vec4(samplePos * NOISINESS, time)) + 1.0) / 2.0;
  sample_ = clamp(sample_, 0.0, 1.0);
  float innerIncBorder = INNER_RADIUS + SHEET_THICKNESS;
  float outerIncBorder = OUTER_RADIUS - SHEET_THICKNESS;
  float radius = innerIncBorder + (outerIncBorder - innerIncBorder) * sample_;

  float dist = distance(pos, vec3(0.0, 0.0, 0.0));
  float density = exp(-pow(dist - radius, 2.0) * 05000.0);
  return (inner_color + (outer_color - inner_color) * (radius - innerIncBorder) / (outerIncBorder - innerIncBorder)) * density;
}

vec4 raymarch(vec3 start, vec3 end) {
  // This is the ray marching function. Here, we sample NUM_STEPS points along the vector
  // between start and end. Then, we integrate the resultant densities linearly.
  vec4 retn = vec4(0.0, 0.0, 0.0, 0.0);
  vec3 delta = end - start;
  float stepDistance = length(delta) / float(NUM_STEPS);

  vec4 densityPrevious = getdensity(start);
  for (int i = 1; i < NUM_STEPS; i++) {
    vec3 samplePos = start + delta * float(i) / float(NUM_STEPS);
    vec4 density = getdensity(samplePos);
    // Integrate the density using linear interpolation
    // The colours will be the average of the two weighted by their alpha
    vec4 densityIntegrated = (density + densityPrevious) / 2.0;
    // Optimised out to return. densityIntegrated *= stepDistance
    retn += densityIntegrated;

    densityPrevious = density;
  }

  return retn * stepDistance;
}

vec4 raymarch_ball(vec2 coord) {
  // Now we're going to intersect a ray from the 
  // coord along the Z axis onto two spheres, one 
  // inside the other (same origin). getdensity 
  // is only > 0 between these volumes.
  float d = distance(coord, vec2(0.0, 0.0));
  if (d > OUTER_RADIUS) {
    // No intersection on the spheres.
    return vec4(0.0, 0.0, 0.0, 0.0);
  }
  float dOuterNormalized = d / OUTER_RADIUS;
  float outerStartZ = -sqrt(1.0 - dOuterNormalized * dOuterNormalized) * OUTER_RADIUS; // sqrt(1-x*x) = function of a circle :)
  float outerEndZ = -outerStartZ;
  if (d > INNER_RADIUS) {
    // The ray only intersects the larger sphere, 
    // so we need to cast from the front to the back

    // We do it twice so that the number of samples in this branch
    // is identical to the number of samples 
    // inside the blob. Otherwise we see artifacts with 
    // a lower number of samples.
    vec4 frontPart = raymarch(vec3(coord, outerStartZ), vec3(coord, 0));
    vec4 backPart = raymarch(vec3(coord, 0), vec3(coord, outerEndZ));
    return frontPart + backPart;
  }

  float dInnerNormalized = d / INNER_RADIUS;
  float innerStartZ = -sqrt(1.0 - dInnerNormalized * dInnerNormalized) * INNER_RADIUS; // sqrt(1-x*x) = function of a circle :)
  float innerEndZ = -innerStartZ;
  // The ray intersects both spheres.
  vec4 frontPart = raymarch(vec3(coord, outerStartZ), vec3(coord, innerStartZ));
  vec4 backPart = raymarch(vec3(coord, innerEndZ), vec3(coord, outerEndZ));
  vec4 final = frontPart + backPart;
  return final;
}

#define PI 3.14159265359

// Simplex noise function
float random(vec2 st) {
  return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

float noise(vec2 st) {
  vec2 i = floor(st);
  vec2 f = fract(st);
  float a = random(i);
  float b = random(i + vec2(1.0, 0.0));
  float c = random(i + vec2(0.0, 1.0));
  float d = random(i + vec2(1.0, 1.0));
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float generateShape(vec2 uv, float seed, float time, float maxRadius) {
  float alpha = 0.0;
  const float mz = 180.0;
  const float step = 1.0;
  float blurAmount = 0.2;
  for (float i = 0.0; i < mz; i += step) {
    float ang = ((noise(vec2(seed, uv.x + i / 200.0) + time / 10.0) * 700.0) * PI) / 180.0;

    vec2 pos = vec2(cos(ang), sin(ang));

    float magn = maxRadius * sin((i * PI) / 100.0);
    pos *= magn;

    vec2 coord = uv + pos / (maxRadius * 10.0);

    alpha += smoothstep(blurAmount, 0.0, length(coord));
  }

  alpha /= (mz / step);
  return min(max(alpha, 0.0), 1.0);
}

void main() {
  float time = uTime * TIME_SCALE;
  vec2 uv = 2.0 * (vTexCoord - 0.5);
  float dist = length(uv);
  vec4 color = vec4(0.0);
  if (dist < INNER_RADIUS) {
    const float step = 1.1;
    float maxRadius = 180.0;

    float seed = 1.;
    const float mz = 180.0;

    color = vec4(0., 0., 0., 1.);

    vec3 c1 = vec3(.8, 1.0, 1.0); // Cyan
    vec3 c2 = vec3(1.0, 1.0, 1.0); // White

    float ct = (cos(3. * uTime) + 1.) / 2.;
    float innerRadius = INNER_RADIUS - 0.05 - 0.01 * (ct);

    float outerRadius = INNER_RADIUS - 0.05 + 0.01 * (ct);
    float blur = .075;

    float color1 = generateShape(uv, (seed), time, innerRadius * maxRadius);
    float color2 = generateShape(uv, (seed + 14.), time, innerRadius * maxRadius);

    float edge1 = smoothstep(outerRadius - blur, outerRadius + blur, dist);
    float edge2 = smoothstep(innerRadius - blur, innerRadius + blur, dist);
    float edge = (edge2 - edge1);
    float b1 = 0.15 + .05 * ct;
    float r1 = .05; //*(sin(time/2.)+1.)/2.;
    float e1 = smoothstep(r1 - b1, r1 + b1, dist);

    vec3 col = mix(c1, c2, (1. - edge));
    col *= mix(c1, c2, e1);

    color.rgb = col;
    color.w = 1.;

    gl_FragColor = vec4(color.rgb, 1.);
    vec4 nc1 = vec4(vec3(.0), .01);
    vec4 nc2 = vec4(vec3(.0), .01);

    float thr = 0.2;

    nc1.b = color1;
    nc2.r = (color2 - color2 * color1);
    // Adjusting mix factors for blurry and faint effect
    vec4 mixedColor1 = mix(color, nc1, nc1.b);
    vec4 mixedColor2 = mix(color, nc2, nc2.r);

    // Adjusting transparency for faint effect
    float faintFactor = .3; // Adjust transparency factor as needed
    vec4 mixedColor = mix(mixedColor1, mixedColor2, 0.5);
    vec4 nc = mix(color, mixedColor, faintFactor);

    gl_FragColor = nc;
    gl_FragColor.w = 1.;
  } else {
    vec4 c5 = raymarch_ball(uv);
    gl_FragColor = c5;
    gl_FragColor.w = length(c5 * c5);
  }

}

`
const vertexShader = `
varying vec2 vTexCoord;

void main() {
    vTexCoord = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

const Bubble: React.FC<{
  state: VoiceModeState
  frequencies: Float32Array[]
}> = ({ state, frequencies }) => {
  const materialRef = useRef<THREE.ShaderMaterial | null>(null)

  const [{ innerRadius, outerRadius, sheetThickness, noisiness, overallScale, timeScale }, set] =
    useControls(() => ({
      innerRadius: { value: 0.25, min: 0, max: 1, step: 0.01 },
      outerRadius: { value: 0.38, min: 0, max: 1, step: 0.01 },
      sheetThickness: { value: 0.02, min: 0, max: 0.08, step: 0.001 },
      noisiness: { value: 6, min: 0, max: 10, step: 0.1 },
      timeScale: { value: 5, min: 0, max: 10, step: 0.1 },
      overallScale: { value: 1, min: 0, max: 5, step: 0.1 },
      idle: button(() => applyPreset('idle')),
      thinking: button(() => applyPreset('thinking')),
      listening: button(() => applyPreset('listening')),
      speaking: button(() => applyPreset('speaking')),
      initializing: button(() => applyPreset('initializing')),
    }))

  const [scaleFactor, setScaleFactor] = useState(1)
  const [frequencyEffect, setFrequencyEffect] = useState(0)

  const applyPreset = useCallback((preset: string) => {
    set(presetStates[preset as keyof typeof presetStates])
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    applyPreset(state)
  }, [state, applyPreset])

  useEffect(() => {
    const summedFrequencies = frequencies.map((bandFrequencies) => {
      const sum = bandFrequencies.reduce((a, b) => a + b, 0)
      return sum / bandFrequencies.length
    })
    if (summedFrequencies.length > 0) {
      setFrequencyEffect(summedFrequencies[0])
    }
  }, [frequencies])

  const { viewport } = useThree()
  useFrame(({ clock }) => {
    ;(THREE.ColorManagement as any).legacyMode = true
    const f = 0.1
    if (materialRef?.current?.uniforms) {
      materialRef.current.uniforms.uTime.value = clock.getElapsedTime()

      const innerRadiusEffect = innerRadius * (1 + frequencyEffect * 0.2)
      const outerRadiusEffect = outerRadius * (1 + frequencyEffect * 0.2)
      const noisinessEffect = noisiness * (1 + frequencyEffect * 0.5)
      const scaleEffect = overallScale * (1 + frequencyEffect * 0.1)

      materialRef.current.uniforms.INNER_RADIUS.value = lerp(
        materialRef.current.uniforms.INNER_RADIUS.value,
        innerRadiusEffect,
        f
      )
      materialRef.current.uniforms.OUTER_RADIUS.value = lerp(
        materialRef.current.uniforms.OUTER_RADIUS.value,
        outerRadiusEffect,
        f
      )

      materialRef.current.uniforms.SHEET_THICKNESS.value = lerp(
        materialRef.current.uniforms.SHEET_THICKNESS.value,
        sheetThickness,
        f
      )
      materialRef.current.uniforms.NOISINESS.value = lerp(
        materialRef.current.uniforms.NOISINESS.value,
        noisinessEffect,
        f
      )
      materialRef.current.uniforms.TIME_SCALE.value = timeScale

      setScaleFactor(lerp(scaleFactor, scaleEffect, 0.1))
    }
  })

  const sMat = useMemo(
    () => (
      <shaderMaterial
        toneMapped
        ref={materialRef}
        transparent
        opacity={2}
        side={2}
        uniforms={{
          uTime: { value: 0 },
          INNER_RADIUS: { value: 0.25 },
          OUTER_RADIUS: { value: 0.37 },
          SHEET_THICKNESS: { value: 0.02 },
          NOISINESS: { value: 6 },
          TIME_SCALE: { value: 5 },
        }}
        fragmentShader={fragmentShader}
        vertexShader={vertexShader}
      />
    ),
    []
  )
  const size = Math.min(viewport.width, viewport.height) * scaleFactor

  return (
    <mesh scale={[size, size, 1]} position={[0, 0, 0]}>
      <planeGeometry args={[1, 1]} />
      {sMat}
    </mesh>
  )
}

export const BubbleAnimation: React.FC<{
  state: VoiceModeState
  frequencies: Float32Array[]
}> = ({ state, frequencies }) => (
  <div className="max-w-[600px] w-full">
    <Canvas
      style={{
        width: '100%',
        height: '100dvh',
        filter: 'saturate(2.5)',
        background: 'white',
      }}
    >
      <Bubble state={state} frequencies={frequencies} />
    </Canvas>
  </div>
)
