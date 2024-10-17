let globalNodes = null, globalEdges = null, input = null, mst = null;
let originalNetwork = null;  // Store the original graph network
let isValidGraphGenerated = false;  // Flag to check if a valid graph has been generated
let shortestPathNetwork = null;

const generateButton = document.querySelector('#generateButton');
const eulerButton = document.querySelector('#eulerButton');
const calculateConnectivityButton = document.querySelector('#calculateConnectivity');
const generateMSTButton = document.querySelector('#generateMST');
const shortestPathButton = document.querySelector('#shortestPathButton');
const inputBox = document.getElementById('degree-sequence');
const sourceVertexInput = document.getElementById('sourceVertex');
const resultMessage = document.getElementById('result-message');
const mstInfo = document.getElementById('mst-info');
const shortestPathInfo = document.getElementById('shortest-path-info');

// Havel-Hakimi Algorithm: Check if a sequence is graphical and build the graph
function havelHakimi(sequence) {
    sequence.sort((a, b) => b - a);  // Sort sequence in descending order

    while (sequence.length > 0) {
        // Remove the first element (largest degree)
        let degree = sequence.shift();

        if (degree > sequence.length) {
            return false;  // Invalid degree sequence
        }

        // Connect this node to 'degree' number of nodes with the highest remaining degrees
        for (let i = 0; i < degree; i++) {
            sequence[i]--;

            // If any degree becomes negative, return false
            if (sequence[i] < 0) {
                return false;
            }
        }

        // Remove all nodes with degree 0
        sequence = sequence.filter(d => d > 0);

        sequence.sort((a, b) => b - a);  // Sort again after modifications
    }

    return true;
}

// Function to determine if the graph is Eulerian
function isEulerianGraph(degrees) {
    const oddDegreeCount = degrees.filter(deg => deg % 2 !== 0).length;
    if (oddDegreeCount === 0) return 2;   // Euler circuit
    if (oddDegreeCount === 2) return 1;   // Euler path
    return 0;  // Neither Euler circuit nor path
}

// Function to generate graph based on input sequence
function generateGraph(sequence) {
    const resultMessage = document.getElementById('result-message');
    const graphContainer = document.getElementById('graph-container');

    // Clear previous graph
    graphContainer.innerHTML = '';

    sequence.sort((a, b) => b - a);
    const pairsOg = sequence.map((val, idx) => {
        return [idx, val];
    });
    let pairs = [...pairsOg];

    const edges = [];
    const nodes = pairs.map(pair => {
        return { id: pair[0], label: `${pair[0]}` };
    });

    while (pairs.length > 0) {
        pairs.sort((a, b) => b[1] - a[1]); // Sort in descending order
        const top = pairs.shift();     // Get the largest degree

        // Create edges
        for (let i = 0; i < top[1]; i++) {
            pairs[i][1]--; // Decrease the degree of the next vertices
            edges.push({ 
                from: top[0], 
                to: pairs[i][0],
                weight: Math.floor(Math.random() * 10) + 1 // Random weight between 1 and 10
            });
            console.log({ from: top[0], to: pairs[i][0], weight: edges[edges.length-1].weight });
        }
        pairs = pairs.filter(pair => pair[1] > 0);
    }

    console.log('edges:');
    console.table(edges);

    const visEdges = edges.map(edge => ({
        ...edge,
        label: edge.weight.toString()
    }));
    const visNodes = nodes;

    // Draw the graph
    const container = graphContainer;  // Use the graph container
    const data = {
        nodes: visNodes,
        edges: visEdges
    };

    const options = {
        physics: false,
        layout: {
            hierarchical: false
        }
    };

    // Create and render the network
    const network = new vis.Network(container, data, options);

    return { nodes, edges };
}

// New function to clear previous outputs
function clearOutputs() {
    resultMessage.textContent = '';
    mstInfo.innerHTML = '';
    shortestPathInfo.textContent = '';
    // Don't clear the graph container here
}

// Update generateButtonPress function
function generateButtonPress() {
    // Clear all previous outputs
    clearAllOutputs();
    isValidGraphGenerated = false;  // Reset the flag

    input = inputBox.value.split(',').map(Number);

    const hasNan = input.find(isNaN);
    if (hasNan !== undefined) {
        resultMessage.textContent = 'Error occurred. Is your input in valid format?';
        return;
    }

    const isGraphical = havelHakimi([...input]);

    if (!isGraphical) {
        resultMessage.textContent = 'Not a valid graphical sequence.';
        return;
    }

    resultMessage.textContent = 'Yes, this is a graphical sequence.';

    let generation = generateGraph(input);
    globalNodes = generation.nodes;
    globalEdges = generation.edges;

    // Display the generated graph
    const container = document.getElementById('graph-container');
    container.innerHTML = '';
    
    const graphContainer = document.createElement('div');
    graphContainer.style.width = '100%';
    graphContainer.style.height = '500px';
    container.appendChild(graphContainer);

    const data = {
        nodes: globalNodes,
        edges: globalEdges.map(edge => ({
            from: edge.from,
            to: edge.to,
            label: edge.weight.toString()
        }))
    };

    const options = {
        physics: false,
        layout: {
            randomSeed: 1  // Use a fixed seed for consistent layouts
        }
    };

    originalNetwork = new vis.Network(graphContainer, data, options);
    isValidGraphGenerated = true;  // Set the flag to true
}

function fleuryAlgorithm(edges, startVertex) {
    const adjacencyList = new Map();
    console.log(edges);
    // Build adjacency list
    for (const [u, v] of edges) {
        if (!adjacencyList.has(u)) adjacencyList.set(u, new Set());
        if (!adjacencyList.has(v)) adjacencyList.set(v, new Set());
        adjacencyList.get(u).add(v);
        adjacencyList.get(v).add(u);
    }

    function dfs(v, visited) {
        visited.add(v);
        for (const neighbor of adjacencyList.get(v)) {
            if (!visited.has(neighbor)) {
                dfs(neighbor, visited);
            }
        }
    }

    function isValidNextEdge(u, v) {
        // If there's only one adjacent vertex, it's always valid
        if (adjacencyList.get(u).size === 1) {
            return true;
        }

        // Count reachable vertices
        const visited1 = new Set();
        dfs(u, visited1);
        const count1 = visited1.size;

        // Remove the edge and count reachable vertices again
        adjacencyList.get(u).delete(v);
        adjacencyList.get(v).delete(u);
        const visited2 = new Set();
        dfs(u, visited2);
        const count2 = visited2.size;

        // Add the edge back
        adjacencyList.get(u).add(v);
        adjacencyList.get(v).add(u);

        // The edge is valid if removing it doesn't disconnect the graph
        return count1 <= count2;
    }

    const circuit = [];
    let currentVertex = startVertex;

    while (adjacencyList.get(currentVertex).size > 0) {
        for (const neighbor of adjacencyList.get(currentVertex)) {
            if (isValidNextEdge(currentVertex, neighbor)) {
                circuit.push(currentVertex);
                adjacencyList.get(currentVertex).delete(neighbor);
                adjacencyList.get(neighbor).delete(currentVertex);
                currentVertex = neighbor;
                break;
            }
        }
    }
    circuit.push(currentVertex);  // Add the last vertex

    return circuit;
}

// Update eulerButtonPress function
function eulerButtonPress() {
    if (globalEdges === null || globalNodes === null || input === null) {
        return resultMessage.textContent = 'First enter valid graphic sequence.';
    }

    const eulerStatus = isEulerianGraph(input);
    if (eulerStatus === 0) {
        return resultMessage.textContent = 'Neither Euler Path nor Euler Circuit is possible';
    }

    // Convert globalEdges to the format expected by fleuryAlgorithm
    const edges = globalEdges.map(edge => [edge.from, edge.to]);

    if (eulerStatus === 1) {
        const startVertex = input.findIndex(deg => deg % 2 !== 0);
        const eulerPath = fleuryAlgorithm(edges, startVertex);
        console.log('Euler Path:', eulerPath.join('->'));
        resultMessage.textContent = `Euler Path is possible (2 odd degree vertices): ${eulerPath.join('->')}`;
    }
    else if (eulerStatus === 2) {
        const eulerCircuit = fleuryAlgorithm(edges, 0);
        console.log('Euler Circuit:', eulerCircuit.join('->'));
        resultMessage.textContent = `Euler circuit is possible (Only even degree vertices): ${eulerCircuit.join('->')}`;
    }
    else {
        console.log('This should not happen');
    }
}

// Update calculateConnectivity function
function calculateConnectivity() {
    if (!isValidGraphGenerated) {
        resultMessage.textContent = 'Please enter a valid graphic sequence and generate a graph first.';
        return;
    }

    clearOutputs();
    if (globalEdges === null || globalNodes === null || input === null) {
        return resultMessage.textContent = 'First enter valid graphic sequence and generate the graph.';
    }

    const edgeConnectivity = calculateEdgeConnectivity();
    const vertexConnectivity = calculateVertexConnectivity();

    let kValues = [];
    for (let i = 0; i<=vertexConnectivity; i++){
        kValues.push(i);
    }
    resultMessage.textContent = `Edge Connectivity: ${edgeConnectivity}, Vertex Connectivity: ${vertexConnectivity}. It's a K-connected graph where K can be ${kValues}`;
}

function calculateEdgeConnectivity() {
    const n = globalNodes.length;
    let minCut = Infinity;

    // Create adjacency matrix
    const graph = Array(n).fill().map(() => Array(n).fill(0));
    for (const edge of globalEdges) {
        graph[edge.from][edge.to] = 1;
        graph[edge.to][edge.from] = 1;
    }

    // Helper function for Ford-Fulkerson algorithm
    function bfs(rGraph, s, t, parent) {
        const visited = Array(n).fill(false);
        const queue = [s];
        visited[s] = true;
        parent[s] = -1;

        while (queue.length) {
            const u = queue.shift();
            for (let v = 0; v < n; v++) {
                if (!visited[v] && rGraph[u][v] > 0) {
                    queue.push(v);
                    parent[v] = u;
                    visited[v] = true;
                }
            }
        }
        return visited[t];
    }

    // Ford-Fulkerson algorithm
    function maxFlow(graph, s, t) {
        const rGraph = graph.map(row => [...row]);
        const parent = Array(n).fill(-1);
        let maxFlow = 0;

        while (bfs(rGraph, s, t, parent)) {
            let pathFlow = Infinity;
            for (let v = t; v !== s; v = parent[v]) {
                const u = parent[v];
                pathFlow = Math.min(pathFlow, rGraph[u][v]);
            }
            for (let v = t; v !== s; v = parent[v]) {
                const u = parent[v];
                rGraph[u][v] -= pathFlow;
                rGraph[v][u] += pathFlow;
            }
            maxFlow += pathFlow;
        }
        return maxFlow;
    }

    // Calculate min-cut (edge connectivity) using max-flow
    for (let i = 0; i < n - 1; i++) {
        for (let j = i + 1; j < n; j++) {
            const flow = maxFlow(graph, i, j);
            minCut = Math.min(minCut, flow);
        }
    }

    return minCut;
}

function calculateVertexConnectivity() {
    const n = globalNodes.length;
    if (n <= 1) return 0;
    if (n === 2) return globalEdges.length > 0 ? 1 : 0;

    let k = 0;
    while (true) {
        if (isKConnected(k + 1)) {
            k++;
        } else {
            break;
        }
    }
    return k;
}

function isKConnected(k) {
    const n = globalNodes.length;
    if (k >= n) return false;

    // Create adjacency list
    const adj = Array(n).fill().map(() => []);
    for (const edge of globalEdges) {
        adj[edge.from].push(edge.to);
        adj[edge.to].push(edge.from);
    }

    // Helper function for DFS
    function dfs(v, visited) {
        visited[v] = true;
        for (const u of adj[v]) {
            if (!visited[u]) {
                dfs(u, visited);
            }
        }
    }

    // Check connectivity after removing each combination of k-1 vertices
    const combinations = getCombinations(n, k - 1);
    for (const removed of combinations) {
        const visited = Array(n).fill(false);
        for (const v of removed) {
            visited[v] = true;
        }

        let start = 0;
        while (start < n && visited[start]) start++;
        if (start === n) continue;

        dfs(start, visited);

        if (visited.some(v => !v)) {
            return false;
        }
    }
    return true;
}

function getCombinations(n, k) {
    const result = [];
    const combination = Array(k).fill(0);

    function generate(start, depth) {
        if (depth === k) {
            result.push([...combination]);
            return;
        }
        for (let i = start; i < n; i++) {
            combination[depth] = i;
            generate(i + 1, depth + 1);
        }
    }

    generate(0, 0);
    return result;
}

// Update generateMST function
function generateMST() {
    if (!isValidGraphGenerated) {
        resultMessage.textContent = 'Please enter a valid graphic sequence and generate a graph first.';
        return;
    }

    // Clear only the MST-related outputs
    clearMSTOutputs();

    // Kruskal's algorithm
    const edges = globalEdges.map(edge => ({
        from: edge.from,
        to: edge.to,
        weight: edge.weight
    })).sort((a, b) => a.weight - b.weight);

    const parent = {};
    const rank = {};

    function makeSet(v) {
        parent[v] = v;
        rank[v] = 0;
    }

    function find(v) {
        if (parent[v] !== v) {
            parent[v] = find(parent[v]);
        }
        return parent[v];
    }

    function union(u, v) {
        const rootU = find(u);
        const rootV = find(v);
        if (rootU !== rootV) {
            if (rank[rootU] < rank[rootV]) {
                parent[rootU] = rootV;
            } else if (rank[rootU] > rank[rootV]) {
                parent[rootV] = rootU;
            } else {
                parent[rootV] = rootU;
                rank[rootU]++;
            }
        }
    }

    globalNodes.forEach(node => makeSet(node.id));

    mst = [];
    edges.forEach(edge => {
        if (find(edge.from) !== find(edge.to)) {
            mst.push(edge);
            union(edge.from, edge.to);
        }
    });

    // Clear the existing graph container
    const container = document.getElementById('graph-container');
    container.innerHTML = '';
    container.style.display = 'flex';
    
    const originalGraphContainer = document.createElement('div');
    originalGraphContainer.style.width = '50%';
    originalGraphContainer.style.height = '500px';
    
    const mstContainer = document.createElement('div');
    mstContainer.style.width = '50%';
    mstContainer.style.height = '500px';
    
    container.appendChild(originalGraphContainer);
    container.appendChild(mstContainer);

    // Recreate original graph
    const originalData = {
        nodes: globalNodes,
        edges: globalEdges.map(edge => ({
            from: edge.from,
            to: edge.to,
            label: edge.weight.toString()
        }))
    };

    const options = {
        physics: false,
        layout: {
            randomSeed: 1  // Use the same seed as in generateButtonPress
        }
    };

    // Render original graph
    new vis.Network(originalGraphContainer, originalData, options);

    // Create MST graph with the same layout as the original
    const mstData = {
        nodes: globalNodes,
        edges: mst.map(edge => ({
            from: edge.from,
            to: edge.to,
            label: edge.weight.toString(),
            color: {color: 'red', highlight: 'red'}
        }))
    };

    // Render MST graph
    new vis.Network(mstContainer, mstData, options);

    // Find fundamental cutsets and circuits
    const fundamentalCutsets = findFundamentalCutsets();
    const fundamentalCircuits = findFundamentalCircuits();

    // Display results
    mstInfo.innerHTML = `
        <h3>Minimum Spanning Tree Weight: ${mst.reduce((sum, edge) => sum + edge.weight, 0)}</h3>
        <h4>Fundamental Cutsets:</h4>
        <ul>${fundamentalCutsets.map(cutset => `<li>${cutset}</li>`).join('')}</ul>
        <h4>Fundamental Circuits:</h4>
        <ul>${fundamentalCircuits.map(circuit => `<li>${circuit}</li>`).join('')}</ul>
    `;
}

function findFundamentalCutsets() {
    const cutsets = [];
    mst.forEach(edge => {
        const cutset = [edge];
        globalEdges.forEach(e => {
            if (!mst.includes(e) && connectsSameComponents(edge, e)) {
                cutset.push(e);
            }
        });
        cutsets.push(cutset);
    });
    return cutsets.map(cutset => 
        cutset.map(edge => `(${edge.from}-${edge.to}, weight: ${edge.weight})`).join(', ')
    );
}

function connectsSameComponents(mstEdge, edge) {
    const mstSet = new Set();
    const queue = [mstEdge.from];
    while (queue.length > 0) {
        const node = queue.shift();
        if (!mstSet.has(node)) {
            mstSet.add(node);
            mst.forEach(e => {
                if (e !== mstEdge) {
                    if (e.from === node) queue.push(e.to);
                    if (e.to === node) queue.push(e.from);
                }
            });
        }
    }
    return (mstSet.has(edge.from) !== mstSet.has(edge.to));
}

function findFundamentalCircuits() {
    const circuits = [];
    globalEdges.forEach(edge => {
        if (!mst.includes(edge)) {
            const circuit = [edge];
            const path = findPath(edge.from, edge.to);
            circuit.push(...path);
            circuits.push(circuit);
        }
    });
    return circuits.map(circuit => 
        circuit.map(edge => `(${edge.from}-${edge.to}, weight: ${edge.weight})`).join(' -> ')
    );
}

function findPath(start, end) {
    const visited = new Set();
    const queue = [[start, []]];
    while (queue.length > 0) {
        const [node, path] = queue.shift();
        if (node === end) return path;
        if (!visited.has(node)) {
            visited.add(node);
            mst.forEach(edge => {
                if (edge.from === node && !visited.has(edge.to)) {
                    queue.push([edge.to, [...path, edge]]);
                }
                if (edge.to === node && !visited.has(edge.from)) {
                    queue.push([edge.from, [...path, edge]]);
                }
            });
        }
    }
    return [];
}

// New function to clear all outputs
function clearAllOutputs() {
    const resultMessage = document.getElementById('result-message');
    const mstInfo = document.getElementById('mst-info');
    const shortestPathInfo = document.getElementById('shortest-path-info');
    const graphContainer = document.getElementById('graph-container');

    if (resultMessage) resultMessage.textContent = '';
    if (mstInfo) mstInfo.innerHTML = '';
    if (shortestPathInfo) shortestPathInfo.textContent = '';
    if (graphContainer) graphContainer.innerHTML = '';
}

// New function to clear only MST-related outputs
function clearMSTOutputs() {
    mstInfo.innerHTML = '';
}

// Dijkstra's algorithm implementation
function dijkstra(graph, source) {
    const distances = {};
    const previous = {};
    const pq = new PriorityQueue();

    // Initialize distances
    for (let node of graph.nodes) {
        distances[node.id] = node.id === source ? 0 : Infinity;
        previous[node.id] = null;
        pq.enqueue(node.id, distances[node.id]);
    }

    while (!pq.isEmpty()) {
        const current = pq.dequeue().element;

        for (let edge of graph.edges.filter(e => e.from === current || e.to === current)) {
            const neighbor = edge.from === current ? edge.to : edge.from;
            const alt = distances[current] + edge.weight;

            if (alt < distances[neighbor]) {
                distances[neighbor] = alt;
                previous[neighbor] = current;
                pq.enqueue(neighbor, alt);
            }
        }
    }

    return { distances, previous };
}

// Helper class for Priority Queue
class PriorityQueue {
    constructor() {
        this.elements = [];
    }

    enqueue(element, priority) {
        this.elements.push({ element, priority });
        this.elements.sort((a, b) => a.priority - b.priority);
    }

    dequeue() {
        return this.elements.shift();
    }

    isEmpty() {
        return this.elements.length === 0;
    }
}

// Function to handle the shortest path button click
function shortestPathButtonPress() {
    if (!isValidGraphGenerated) {
        resultMessage.textContent = 'Please enter a valid graphic sequence and generate a graph first.';
        return;
    }

    const source = parseInt(sourceVertexInput.value);
    if (isNaN(source) || source < 0 || source >= globalNodes.length) {
        resultMessage.textContent = 'Please enter a valid source vertex.';
        return;
    }

    clearOutputs();

    const graph = { nodes: globalNodes, edges: globalEdges };
    const { distances, previous } = dijkstra(graph, source);

    // Prepare the shortest path information
    let pathInfo = `Shortest paths from vertex ${source}:\n`;
    for (let node of globalNodes) {
        if (node.id !== source) {
            const path = getPath(previous, source, node.id);
            pathInfo += `To ${node.id}: ${path.join(' -> ')} (Distance: ${distances[node.id]})\n`;
        }
    }

    // Display the shortest path information
    shortestPathInfo.textContent = pathInfo;

    // Visualize the shortest paths
    visualizeShortestPaths(graph, source, distances, previous);
}

// Helper function to reconstruct the path
function getPath(previous, start, end) {
    const path = [];
    let current = end;
    while (current !== null) {
        path.unshift(current);
        current = previous[current];
    }
    return path;
}

// Function to visualize the shortest paths
function visualizeShortestPaths(graph, source, distances, previous) {
    const container = document.getElementById('graph-container');
    container.innerHTML = '';
    container.style.display = 'flex';
    
    const shortestPathContainer = document.createElement('div');
    shortestPathContainer.style.width = '100%';
    shortestPathContainer.style.height = '500px';
    
    container.appendChild(shortestPathContainer);

    const nodes = graph.nodes.map(node => ({
        ...node,
        color: node.id === source ? '#ff0000' : '#97c2fc',
        label: `${node.id}\n(${distances[node.id]})`
    }));

    const edges = graph.edges.map(edge => {
        const isInPath = previous[edge.to] === edge.from || previous[edge.from] === edge.to;
        return {
            ...edge,
            color: isInPath ? '#ff0000' : '#848484',
            width: isInPath ? 2 : 1,
            label: edge.weight.toString()
        };
    });

    const data = { nodes, edges };
    const options = {
        physics: false,
        layout: {
            randomSeed: 1
        }
    };

    shortestPathNetwork = new vis.Network(shortestPathContainer, data, options);
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    const generateButton = document.querySelector('#generateButton');
    const eulerButton = document.querySelector('#eulerButton');
    const calculateConnectivityButton = document.querySelector('#calculateConnectivity');
    const generateMSTButton = document.querySelector('#generateMST');
    const shortestPathButton = document.querySelector('#shortestPathButton');

    if (generateButton) {
        generateButton.addEventListener('click', generateButtonPress);
    } else {
        console.error('Generate button not found');
    }

    if (eulerButton) {
        eulerButton.addEventListener('click', eulerButtonPress);
    } else {
        console.error('Euler button not found');
    }

    if (calculateConnectivityButton) {
        calculateConnectivityButton.addEventListener('click', calculateConnectivity);
    } else {
        console.error('Calculate Connectivity button not found');
    }

    if (generateMSTButton) {
        generateMSTButton.addEventListener('click', generateMST);
    } else {
        console.error('Generate MST button not found');
    }

    if (shortestPathButton) {
        shortestPathButton.addEventListener('click', shortestPathButtonPress);
    } else {
        console.error('Shortest Path button not found');
    }
});