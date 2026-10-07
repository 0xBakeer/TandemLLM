---
name: threejs-scene
description: Building a WebGL scene with Three.js - renderer, camera, lights, meshes, controls, resize, animation loop, and performance (draw calls, instancing, disposal).
---
# Three.js scene

Rules:
- Import from one source (an ES module import map or a bundler), one version. Never mix a CDN build and a bundled one.
- Order: renderer with `antialias`, `setPixelRatio(min(dpr, 2))`, `setSize`; a `PerspectiveCamera` placed to see the subject; one ambient or hemisphere light plus one directional light; then meshes.
- Camera first. If nothing shows, the camera is inside, behind, or too far: print `camera.position` and the object's bounding box.
- Reuse geometries and materials. Many identical objects: `InstancedMesh`. Target under 1,000 draw calls (`renderer.info.render.calls`).

What goes wrong:
- A black canvas: no light, a material that needs light (`MeshStandardMaterial`) with none, or the camera not looking at the object.
- Shaking or z-fighting: `near` too small (use 0.1) or `far` too large for the scene scale.
- Textures black: loaded after the first render with no `needsUpdate`, or a CORS block; read the console.

Check: run 3 s, read `renderer.info.render` (calls, triangles) and the measured fps; screenshot at 1280 and 390; console 0 errors.

Done when:
- [ ] Screenshot shows the lit subject centered at both widths.
- [ ] Draw calls, triangles and fps read from `renderer.info` (through the state hook; the delivered page shows no fps or stats readout).
- [ ] Orbit or key input through the browser tool moves the camera; position read back.

## Reference

Import map (no bundler):
```html
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.170.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.170.0/examples/jsm/"}}</script>
```
Minimal scene:
```js
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x111318);
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 200);
camera.position.set(4, 3, 6); camera.lookAt(0, 0, 0);
scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 0.8));
const sun = new THREE.DirectionalLight(0xffffff, 1.2); sun.position.set(5, 10, 4); scene.add(sun);
const cube = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial({ color: 0x3b82f6 }));
scene.add(cube);
const controls = new OrbitControls(camera, renderer.domElement);
addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
let frames = 0;
renderer.setAnimationLoop(() => { frames++; cube.rotation.y += 0.01; controls.update(); renderer.render(scene, camera); });
window.__state = () => ({ tandem_state: 1, frames, calls: renderer.info.render.calls, tris: renderer.info.render.triangles, cam: camera.position.toArray() });
```
Instancing: `const m = new THREE.InstancedMesh(geo, mat, n); m.setMatrixAt(i, matrix); m.instanceMatrix.needsUpdate = true;`
Disposal: `mesh.geometry.dispose(); mesh.material.dispose(); scene.remove(mesh);`

More rules:
- Resize handler updates `camera.aspect`, `updateProjectionMatrix()`, `renderer.setSize`.
- Dispose geometries, materials, textures when removed; otherwise memory grows.

More pitfalls:
- Frame rate collapses: shadows on every light, per-frame `new` objects, thousands of meshes instead of one instanced mesh.
- Controls do nothing: `controls.update()` missing in the loop, or the renderer canvas is behind another element.
