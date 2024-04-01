import * as THREE from 'three';
import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

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
let camera2 = new THREE.PerspectiveCamera(30, window.innerWidth / (window.innerHeight * 0.5), 0.1, 1000);

// Initialize controls1 here to avoid duplicates
let controls2 = new OrbitControls(camera2, renderer2.domElement);

// Initialize stats2, gui2, and set isInitialized2 flag
let stats2 = new Stats();
let gui2 = new GUI();
let isInitialized2 = false;


// Container3
const container3 = document.getElementById('container3');
container3.style.position = 'relative';

// Initialize the renderer3, scene3, and camera3
let renderer3 = new THREE.WebGLRenderer();
let scene3 = new THREE.Scene();
let camera3 = new THREE.PerspectiveCamera(30, window.innerWidth / (window.innerHeight * 0.5), 0.1, 1000);

// Initialize controls1 here to avoid duplicates
let controls3 = new OrbitControls(camera3, renderer3.domElement);

// Initialize stats3, gui3, and set isInitialized3 flag
let stats3 = new Stats();
let gui3 = new GUI();
let isInitialized3 = false;


function initScene(container, renderer, scene, camera, controls) {
	scene.background = new THREE.Color(0xffffff);
	renderer.setSize(window.innerWidth, window.innerHeight * 0.5);
	container.appendChild(renderer.domElement);

	camera.position.z = 25;

	let dirlight = new THREE.DirectionalLight(0xffffff, 0.5);
	dirlight.position.set(0, 0, 1);
	scene.add(dirlight);

	let ambientLight = new THREE.AmbientLight(0x404040, 2);
	scene.add(ambientLight);
}

function loadObject(scene, objectName, objectPath, container) {
	let loader = new PLYLoader(); // Assuming PLYLoader is already imported
	loader.load(
		objectPath,
		function (geometry) {
			// Create a material for the points
			let material = new THREE.PointsMaterial({
				color: 0x999999,
				size: 0.1, // Adjust the size of the points
				// sizeAttenuation: true // Uncomment if you want the points to attenuate (shrink) with distance
			});

			// Create a points object instead of a mesh
			let points = new THREE.Points(geometry, material);
			points.position.set(0, 0, 0);
			points.name = objectName;
			scene.add(points);

			// Determine which container and flags to use
			if (container === container1 && !isInitialized1) {
				initGUI(container1, gui1, scene, points);
				isInitialized1 = true;
			} else if (container === container2 && !isInitialized2) {
				initGUI(container2, gui2, scene, points);
				isInitialized2 = true;
			} else if (container === container3 && !isInitialized3) {
				initGUI(container3, gui3, scene, points);
				isInitialized3 = true;
			}
		},
		function (xhr) {
			console.log((xhr.loaded / xhr.total * 100) + '% loaded');
		},
		function (error) {
			console.log('An error happened: ' + error);
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

function initGUI(container, gui, scene, plots) {
	gui.add(plots.position, 'x', -1, 1);
	gui.add(plots.position, 'y', -1, 1);
	gui.add(plots.position, 'z', -1, 1);
	gui.domElement.style.position = 'absolute';
	gui.domElement.style.top = '0px';
	gui.domElement.style.right = '0px';
	container.appendChild(gui.domElement);
}

function animate(renderer, scene, camera, stats) {
	requestAnimationFrame(() => animate(renderer, scene, camera, stats));

	let plots = scene.getObjectByName("plots");
	if (plots) {
		plots.rotation.x += 0.005;
		plots.rotation.y += 0.005;
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
	onWindowResize(camera2, renderer2);
	onWindowResize(camera3, renderer3);
}, false);

// Initialize the scene, stats, and start the animation loop - container 1
initScene(container1, renderer1, scene1, camera1, controls1);
initSTATS(container1, stats1);
loadObject(scene1, "ply1", "../assets/results/assignment2/backup/sfm_output.ply", container1);
animate(renderer1, scene1, camera1, stats1);

// Initialize the scene, stats, and start the animation loop - container 2
initScene(container2, renderer2, scene2, camera2, controls2);
initSTATS(container2, stats2);
loadObject(scene2, "ply2", "../assets/results/assignment2/backup/colmap_output.ply", container2);
animate(renderer2, scene2, camera2, stats2);

// Initialize the scene, stats, and start the animation loop - container 3
initScene(container3, renderer3, scene3, camera3, controls3);
initSTATS(container3, stats3);
loadObject(scene3, "ply3", "../assets/results/assignment2/backup/ec_output.ply", container3);
animate(renderer3, scene3, camera3, stats3);
