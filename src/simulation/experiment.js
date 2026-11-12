import { saveJsonFile } from '../utils/temporalLog';

const TRIPS = 3;
const MIN_VEHICLES = 4;
const MAX_VEHICLES = 60;

// Each chain is three linked trips. Every leg is two to four intersections
// and has another first road within two extra hops, so a queue on the short
// road can be avoided. Chains share the short downtown and hub roads, and
// the list repeats as the vehicle count grows.
const FLOW_ZONES = [
  [0, 3, 5, 10],
  [1, 13, 5, 12],
  [2, 5, 10, 12],
  [3, 4, 2, 13],
  [4, 3, 5, 10],
  [5, 2, 7, 13],
  [6, 4, 2, 12],
  [7, 9, 2, 13],
  [8, 0, 2, 13],
  [9, 1, 7, 13],
  [10, 4, 2, 12],
  [11, 4, 2, 12],
  [12, 1, 7, 13],
  [13, 4, 2, 12],
  [12, 2, 7, 13],
  [13, 5, 2, 12],
  [0, 3, 13, 4],
  [1, 13, 6, 12]
];

// Roads the training batches take turns filling. Low-capacity downtown links
// sit next to the busier hub roads.
const BOTTLENECKS = [
  [9, 12], [10, 13], [13, 10], [5, 9], [9, 5], [1, 5],
  [0, 1], [1, 2], [6, 10], [10, 6], [2, 6], [6, 2],
  [5, 6], [6, 5], [4, 13], [3, 9]
];

export function clampTripCount(count) {
  const n = Number(count);
  if (!Number.isFinite(n)) return 24;
  return Math.max(MIN_VEHICLES, Math.min(MAX_VEHICLES, Math.round(n)));
}

function trialSteps(count) {
  return Math.min(8000, 2800 + Math.max(0, count - 8) * 90);
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return function rand() {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function yieldThread() {
  return new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
}

function pickSpeed(rand) {
  const roll = rand();
  if (roll < 0.3) return 2.2 + rand();
  if (roll < 0.75) return 4.4 + rand() * 1.4;
  return 7.2 + rand() * 1.8;
}

function firstHopAlternate(sim, from, to, shortest) {
  const maxHops = shortest.length + 1;
  const stack = [[from]];
  while (stack.length > 0) {
    const path = stack.pop();
    if (path.length - 1 > maxHops) continue;
    const at = path[path.length - 1];
    if (at === to && path.length > 1) {
      if (path[1] !== shortest[1]) return true;
      continue;
    }
    const roads = sim.outgoing.get(at) || [];
    for (let r = 0; r < roads.length; r++) {
      const next = roads[r].toZone;
      if (path.indexOf(next) >= 0) continue;
      stack.push(path.concat(next));
    }
  }
  return false;
}

function usableLeg(sim, from, to) {
  const path = sim.shortestPath(from, to);
  if (!path || path.length < 3 || path.length > 5) return null;
  if (!firstHopAlternate(sim, from, to, path)) return null;
  return path;
}

function pathUses(path, edge) {
  for (let i = 0; i < path.length - 1; i++) {
    if (path[i] === edge[0] && path[i + 1] === edge[1]) return true;
  }
  return false;
}

function legsFrom(sim) {
  const bySource = [];
  for (let zone = 0; zone < sim.zones.length; zone++) bySource.push([]);
  for (let from = 0; from < sim.zones.length; from++) {
    for (let to = 0; to < sim.zones.length; to++) {
      const path = usableLeg(sim, from, to);
      if (path) bySource[from].push({ dest: to, path });
    }
  }
  return bySource;
}

function chainToTrips(sim, chain) {
  const trips = [];
  for (let i = 0; i < chain.length - 1; i++) {
    if (!usableLeg(sim, chain[i], chain[i + 1])) return null;
    trips.push({ source: chain[i], dest: chain[i + 1] });
  }
  return trips.length === TRIPS ? trips : null;
}

function shuffle(list, rand) {
  for (let i = list.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const swap = list[i];
    list[i] = list[j];
    list[j] = swap;
  }
  return list;
}

export function buildTripDataset(sim, count, seed) {
  const total = clampTripCount(count);
  const rand = mulberry32(seed + 17);
  const vehicles = [];
  for (let i = 0; i < total; i++) {
    const chain = FLOW_ZONES[i % FLOW_ZONES.length];
    const trips = chainToTrips(sim, chain);
    if (!trips) continue;
    vehicles.push({
      id: vehicles.length + 1,
      speed: Math.round(pickSpeed(rand) * 10) / 10,
      trips
    });
  }
  return { seed, count: vehicles.length, tripsPerVehicle: TRIPS, vehicles };
}

// A new training batch on every click. Starts are spread across the zones.
// The second and third trips of the batch merge onto two bottlenecks, so the
// model sees a queue and the other road around it. The bottlenecks change
// with the seed, so the next click is a different list.
export function buildTrainingDataset(sim, count, seed) {
  const total = clampTripCount(count);
  const rand = mulberry32(seed >>> 0);
  const legs = legsFrom(sim);
  const first = Math.floor(rand() * BOTTLENECKS.length);
  let second = Math.floor(rand() * BOTTLENECKS.length);
  if (second === first) second = (second + 1) % BOTTLENECKS.length;
  const focus = [BOTTLENECKS[first], BOTTLENECKS[second]];

  const starts = [];
  while (starts.length < total) {
    const pass = [];
    for (let zone = 0; zone < sim.zones.length; zone++) pass.push(zone);
    shuffle(pass, rand);
    for (let i = 0; i < pass.length && starts.length < total; i++) starts.push(pass[i]);
  }

  const used = new Set();
  const vehicles = [];
  for (let i = 0; i < total; i++) {
    let source = starts[i];
    const trips = [];
    const seen = new Set([source]);
    let recent = new Set();
    for (let t = 0; t < TRIPS; t++) {
      const from = source;
      const blocked = recent;
      const ahead = legs[from].filter((leg) => !seen.has(leg.dest));
      if (ahead.length === 0) break;
      const noLoop = ahead.filter((leg) => {
        for (let k = 1; k < leg.path.length; k++) {
          if (blocked.has(leg.path[k])) return false;
        }
        return true;
      });
      const noUturn = ahead.filter((leg) => leg.path[1] == null || !blocked.has(leg.path[1]));
      const options = noLoop.length > 0 ? noLoop : (noUturn.length > 0 ? noUturn : ahead);
      let pool = options;
      if (t === 0) {
        const feeders = options.filter((leg) => legs[leg.dest].some((next) => (
          next.dest !== from && pathUses(next.path, focus[0])
        )));
        if (feeders.length > 0) pool = feeders;
      } else {
        const edge = focus[t - 1];
        const through = options.filter((leg) => pathUses(leg.path, edge));
        if (through.length > 0) pool = through;
      }
      const fresh = pool.filter((leg) => !used.has(`${from}-${leg.dest}`));
      if (fresh.length > 0) pool = fresh;
      const leg = pool[Math.floor(rand() * pool.length)];
      used.add(`${from}-${leg.dest}`);
      trips.push({ source: from, dest: leg.dest });
      seen.add(leg.dest);
      recent = new Set(leg.path.slice(0, -1));
      source = leg.dest;
    }
    if (trips.length < TRIPS) continue;
    vehicles.push({
      id: vehicles.length + 1,
      speed: Math.round(pickSpeed(rand) * 10) / 10,
      trips
    });
  }
  return {
    seed,
    count: vehicles.length,
    tripsPerVehicle: TRIPS,
    vehicles,
    focus: focus.map((edge) => `${sim.zones[edge[0]].name} → ${sim.zones[edge[1]].name}`)
  };
}

function namedTrips(sim, trips) {
  return trips.map((trip) => (
    `${sim.zones[trip.source].name} → ${sim.zones[trip.dest].name}`
  )).join(' · ');
}

function namedTrace(sim, trace) {
  if (!trace || trace.length === 0) return '—';
  return trace.map((id) => sim.zones[id].name).join(' → ');
}

async function runTrial(sim, dataset, seed, guide, steps, onProgress, label) {
  sim.rng = mulberry32(seed);
  sim.resetForExperiment();
  for (let i = 0; i < dataset.vehicles.length; i++) {
    sim.spawnTripVehicle(dataset.vehicles[i], guide);
  }
  const frames = [];
  let load = 0;
  let peak = 0;
  let waiting = 0;
  for (let step = 0; step < steps; step++) {
    const sample = sim.experimentAdvance(guide);
    frames.push(sample);
    load += sample.load;
    peak += sample.peak;
    waiting += sample.waiting;
    if (step % 40 === 0) {
      onProgress(`${label} ${step}/${steps}`);
      await yieldThread();
    }
  }
  const journeys = dataset.vehicles.length * TRIPS;
  const done = sim.finished.length;
  const still = sim.vehicles.map((vehicle) => ({
    id: vehicle.id,
    speed: Math.round(vehicle.speed * 10) / 10,
    trips: vehicle.trips,
    trace: vehicle.trace.slice()
  }));
  const records = sim.finished.concat(still).sort((a, b) => a.id - b.id);
  return {
    mapLoad: load / steps,
    peakLoad: peak / steps,
    meanWaiting: waiting / steps,
    finished: done,
    journeys,
    frames,
    vehicles: records.map((record) => ({
      id: record.id,
      plan: namedTrips(sim, record.trips),
      path: namedTrace(sim, record.trace)
    }))
  };
}

function publicResult(result) {
  return {
    mapLoad: round(result.mapLoad),
    peakLoad: round(result.peakLoad),
    meanWaiting: round(result.meanWaiting),
    finished: result.finished,
    journeys: result.journeys,
    vehicles: result.vehicles
  };
}

function round(value) {
  return Math.round(value * 1000) / 1000;
}

function publicDataset(sim, dataset, seed) {
  return {
    seed,
    count: dataset.count,
    tripsPerVehicle: dataset.tripsPerVehicle,
    vehicles: dataset.vehicles.map((vehicle) => ({
      id: vehicle.id,
      speed: vehicle.speed,
      trips: vehicle.trips,
      plan: namedTrips(sim, vehicle.trips)
    }))
  };
}

function restoreForCollect(sim) {
  sim.rng = null;
  sim.resetEnvironment();
  sim.setMode('collect');
}

// Run the saved trip list on a brand-new model. Training stays off so the
// next step can be Add vehicles and Run.
export async function runInitialDatasetTest(sim, { count, onProgress }) {
  const vehicleCount = clampTripCount(count);
  const steps = trialSteps(vehicleCount);
  const seed = 1000 + vehicleCount * 17;
  const report = (text) => {
    if (onProgress) onProgress(text);
  };

  report('Building the fixed trip list');
  const dataset = buildTripDataset(sim, vehicleCount, seed);
  sim.freshModel();
  sim.tripDataset = dataset;
  sim.tripSeed = seed;
  sim.tripSteps = steps;

  let untrained;
  try {
    report('Running the initial model');
    untrained = await runTrial(sim, dataset, seed, 'shortest', steps, report, 'Initial');
  } finally {
    restoreForCollect(sim);
  }

  const result = {
    dataset: publicDataset(sim, dataset, seed),
    steps,
    untrained: publicResult(untrained),
    trained: null,
    loss: null,
    rounds: 0,
    mapChange: null
  };
  sim.tripUntrained = result.untrained;

  const saved = await Promise.all([
    saveJsonFile('trip-dataset.json', result.dataset),
    saveJsonFile('result-untrained.json', result.untrained)
  ]);
  result.saved = saved.every(Boolean);
  report(result.saved
    ? 'Initial result saved. Add vehicles and press Run.'
    : 'Initial result is ready. Add vehicles and press Run.');
  return result;
}

// Run the same trip list on the model trained by Collect mode.
export async function runSameDatasetTest(sim, { onProgress }) {
  if (!sim.tripDataset || !sim.tripUntrained) {
    throw new Error('Test the initial model first.');
  }
  if (sim.stats.rounds < 1) {
    throw new Error('Collect data first. Add vehicles and press Run.');
  }
  const report = (text) => {
    if (onProgress) onProgress(text);
  };

  let trained;
  try {
    report('Running the trained model on the same trips');
    trained = await runTrial(
      sim,
      sim.tripDataset,
      sim.tripSeed,
      'trained',
      sim.tripSteps || trialSteps(sim.tripDataset.count),
      report,
      'Trained'
    );
  } finally {
    restoreForCollect(sim);
  }

  const trainedResult = publicResult(trained);
  const weights = sim.model.getWeights();
  const comparison = {
    dataset: publicDataset(sim, sim.tripDataset, sim.tripSeed),
    steps: sim.tripSteps || trialSteps(sim.tripDataset.count),
    untrained: sim.tripUntrained,
    trained: trainedResult,
    loss: round(sim.stats.loss),
    rounds: sim.stats.rounds,
    mapChange: round(trainedResult.mapLoad - sim.tripUntrained.mapLoad)
  };

  const saved = await Promise.all([
    saveJsonFile('result-trained.json', comparison.trained),
    saveJsonFile('model-trained.json', {
      loss: comparison.loss,
      rounds: comparison.rounds,
      weights
    }),
    saveJsonFile('comparison.json', {
      mapChange: comparison.mapChange,
      untrained: comparison.untrained.mapLoad,
      trained: comparison.trained.mapLoad,
      loss: comparison.loss,
      rounds: comparison.rounds
    })
  ]);
  comparison.saved = saved.every(Boolean);
  report(comparison.saved
    ? 'Saved the trained result and the model.'
    : 'Trained result is ready.');
  return comparison;
}
