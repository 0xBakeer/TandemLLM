---
name: threejs-scene
description: Building a WebGL scene with Three.js - renderer, camera, lights, meshes, controls, resize, animation loop, and performance (draw calls, instancing, disposal).
---
# Three.js scene

Rules:
- One import source (an import map or a bundler), one version.
- Renderer: `antialias`, `setPixelRatio(min(dpr, 2))`, `toneMapping = ACESFilmicToneMapping`; colour textures get `colorSpace = SRGBColorSpace`.
- Light for the mood asked. Day: a sky-blue background, fog of the same colour starting beyond the subject, a `HemisphereLight` (sky, warm ground, ~1.5) and a sun `DirectionalLight` (~3) with shadows. Night is a choice, never the default; a day/night switch changes background, fog, both lights and exposure together.
- Materials: a distinct, saturated colour per part; `metalness: 0` for wood, stone and leaves (metal without an environment map renders black); `flatShading` for low-poly forms. Foliage is clustered icosahedra or cones, never flat squares.
- Camera first: if nothing shows, print `camera.position` and the subject's bounding box. Reuse geometries and materials; `InstancedMesh` for many of one.

Check: screenshot at 1280 and 390 and compare it with the request: daylight reads bright, the subject is recognisable, the colours differ. Console 0 errors; `renderer.info.render.calls` under 1,000.

Done when:
- [ ] The screenshot shows the lit subject centered at both widths, in the light the request names.
- [ ] Draw calls and fps read through the state hook (the page shows no stats readout).
- [ ] Orbit or key input moves the camera; position read back.

## Reference

Import map:
```html
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.170.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.170.0/examples/jsm/"}}</script>
```
A day scene:
```js
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);
const sky = 0x9fd3ff, scene = new THREE.Scene();
scene.background = new THREE.Color(sky); scene.fog = new THREE.Fog(sky, 30, 120);
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 300);
camera.position.set(8, 6, 12);
scene.add(new THREE.HemisphereLight(0xdcefff, 0x8a7350, 1.5));
const sun = new THREE.DirectionalLight(0xfff1d6, 3); sun.position.set(10, 18, 6); sun.castShadow = true; scene.add(sun);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.MeshStandardMaterial({ color: 0x6fa04a }));
ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; scene.add(ground);
const cube = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial({ color: 0xc8402c, metalness: 0 }));
cube.position.y = 0.5; cube.castShadow = true; scene.add(cube);
const controls = new OrbitControls(camera, renderer.domElement);
addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
let frames = 0;
renderer.setAnimationLoop(() => { frames++; controls.update(); renderer.render(scene, camera); });
window.__state = () => ({ tandem_state: 1, frames, calls: renderer.info.render.calls, cam: camera.position.toArray() });
```
Night: background and fog 0x0b1530, hemisphere 0.3, moon 0.8 (0xaac4ff), lanterns as `PointLight`s.

What goes wrong:
- Dull, grey or dark: lights at 1 or less (dim since r155), a dark or grey background, fog nearer than the subject, metalness above 0. Black: no light, or the camera looks away.
- Frame rate collapses: shadows on every light, `new` objects each frame, thousands of meshes instead of an `InstancedMesh`; dispose what you remove.
