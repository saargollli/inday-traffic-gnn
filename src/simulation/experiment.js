import { saveJsonFile } from '../utils/temporalLog';

const TRIPS = 3;
const TRIAL_STEPS = 1000;

// Shared itineraries. Several vehicles are given the same chain so the
// untrained shortest roads pile up, and the trained model has a jam to avoid.
const ITINERARIES = [
  [0, 5, 10, 13],
  [3, 6, 9, 12],
  [8, 9, 6, 2],
  [11, 7, 2, 0],
  [4, 5, 6, 7]
];

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

function pickDest(sim, source, rand) {
  const options = [];
  const fallback = [];
  for (let zone = 0; zone < sim.zones.length; zone++) {
    if (zone === source) continue;
    const path = sim.shortestPath(source, zone);
    if (!path) continue;
    fallback.push(zone);
    if (path.length >= 3 && path.length <= 5) options.push(zone);
  }
  const pool = options.length > 0 ? options : fallback;
  return pool[Math.floor(rand() * pool.length)];
}

export function buildTripDataset(sim, count, seed) {
  const rand = mulberry32(seed + 17);
  const vehicles = [];
  for (let i = 0; i < count; i++) {
    const chain = ITINERARIES[i % ITINERARIES.length];
    const trips = [];
    for (let t = 0; t < chain.length - 1; t++) {
      const source = chain[t];
      const dest = chain[t + 1];
      if (!sim.shortestPath(source, dest)) continue;
      trips.push({ source, dest });
    }
    if (trips.length === 0) {
      let source = Math.floor(rand() * sim.zones.length);
      for (let t = 0; t < TRIPS; t++) {
        const dest = pickDest(sim, source, rand);
        trips.push({ source, dest });
        source = dest;
      }
    }
    vehicles.push({
      id: i + 1,
      speed: Math.round(pickSpeed(rand) * 10) / 10,
      trips
    });
  }
  return { seed, count, tripsPerVehicle: TRIPS, vehicles };
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
  const vehicleCount = Math.max(4, Math.min(20, count || 12));
  const seed = 1000 + vehicleCount * 17;
  const report = (text) => {
    if (onProgress) onProgress(text);
  };

  report('Building the fixed trip list');
  const dataset = buildTripDataset(sim, vehicleCount, seed);
  sim.freshModel();
  sim.tripDataset = dataset;
  sim.tripSeed = seed;

  let untrained;
  try {
    report('Running the initial model');
    untrained = await runTrial(sim, dataset, seed, 'shortest', TRIAL_STEPS, report, 'Initial');
  } finally {
    restoreForCollect(sim);
  }

  const result = {
    dataset: publicDataset(sim, dataset, seed),
    steps: TRIAL_STEPS,
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
      TRIAL_STEPS,
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
    steps: TRIAL_STEPS,
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
