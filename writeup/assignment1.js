import * as THREE from 'three';
import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

// Container1
const container1 = document.getElementById('container1');
container1.style.position = 'relative';

// Initialize the renderer1, scene1, and camera1
let renderer1 = new THREE.WebGLRenderer();
let scene1 = new THREE.Scene();
let camera1 = new THREE.PerspectiveCamera(30, window.innerWidth / (window.innerHeight * 0.5), 0.1, 1000);

// Initialize controls1 here to avoid duplicates
let controls1 = new OrbitControls(camera1, renderer1.domElement);

// Initialize stats1, gui1, and set isInitialized1 flag
let stats1 = new Stats();
let gui1 = new GUI();
let isInitialized1 = false;


// Container2
const container2 = document.getElementById('container2');
container2.style.position = 'relative';

// Initialize the renderer2, scene2, and camera2
let renderer2 = new THREE.WebGLRenderer();
let scene2 = new THREE.Scene();
let camera2 = new THREE.PerspectiveCamera(50, window.innerWidth / (window.innerHeight * 0.5), 0.1, 1000);

// Initialize controls1 here to avoid duplicates
let controls2 = new OrbitControls(camera2, renderer2.domElement);

// Initialize stats2, gui2, and set isInitialized2 flag
let stats2 = new Stats();
let gui2 = new GUI();
let isInitialized2 = false;

function initScene(container, renderer, scene, camera, controls) {
	scene.background = new THREE.Color(0xffffff);
	renderer.setSize(window.innerWidth, window.innerHeight * 0.5);
	container.appendChild(renderer.domElement);

	camera.position.z = 5;

	let dirlight = new THREE.DirectionalLight(0xffffff, 0.5);
	dirlight.position.set(0, 0, 1);
	scene.add(dirlight);

	let ambientLight = new THREE.AmbientLight(0x404040, 2);
	scene.add(ambientLight);
}

function loadObject(scene, objectName, objectPath, container) {
	let loader = new OBJLoader();
	loader.load(
		objectPath,
		function (object) {
			let cube = object.children[0];
			cube.material = new THREE.MeshPhongMaterial({ color: 0x999999 });
			cube.position.set(0, 0, 0);
			cube.name = objectName;
			scene.add(cube);

			// Determine which container and flags to use
			if (container === container1 && !isInitialized1) {
				initGUI(container1, gui1, scene, cube);
				isInitialized1 = true;
			} else if (container === container2 && !isInitialized2) {
				initGUI(container2, gui2, scene, cube);
				isInitialized2 = true;
			}
		},
		function (xhr) {
			console.log((xhr.loaded / xhr.total * 100) + '% loaded');
		},
		function (error) {
			console.log('An error happened' + error);
		}
	);
}

function initSTATS(container, stats) {
	stats.showPanel(0);
	stats.domElement.style.position = 'absolute';
	stats.domElement.style.top = '0';
	stats.domElement.style.left = '0';
	container.appendChild(stats.domElement);
}

function initGUI(container, gui, scene, cube) {
	gui.add(cube.position, 'x', -1, 1);
	gui.add(cube.position, 'y', -1, 1);
	gui.add(cube.position, 'z', -1, 1);
	gui.domElement.style.position = 'absolute';
	gui.domElement.style.top = '0px';
	gui.domElement.style.right = '0px';
	container.appendChild(gui.domElement);
}

function animate(renderer, scene, camera, stats) {
	requestAnimationFrame(() => animate(renderer, scene, camera, stats));

	let cube = scene.getObjectByName("cube");
	if (cube) {
		cube.rotation.x += 0.005;
		cube.rotation.y += 0.005;
	}

	renderer.render(scene, camera);
	stats.update();
}

function onWindowResize(camera, renderer) {
	camera.aspect = window.innerWidth / (window.innerHeight * 0.5);
	camera.updateProjectionMatrix();
	renderer.setSize(window.innerWidth, window.innerHeight * 0.5);
}

window.addEventListener('resize', () => {
	onWindowResize(camera1, renderer1);
	onWindowResize(camera2, renderer2)
}, false);

// Initialize the scene, stats, and start the animation loop - container 1
initScene(container1, renderer1, scene1, camera1, controls1);
initSTATS(container1, stats1);
loadObject(scene1, "cube", "../assets/results/assignment1/cube_subdivided.obj", container1);
animate(renderer1, scene1, camera1, stats1);

// Initialize the scene, stats, and start the animation loop - container 2
initScene(container2, renderer2, scene2, camera2, controls2);
initSTATS(container2, stats2);
loadObject(scene2, "cube", "../assets/results/assignment1/cube_decimated.obj", container2);
animate(renderer2, scene2, camera2, stats2);
