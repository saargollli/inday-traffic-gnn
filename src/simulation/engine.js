import { TemporalGNN, FEATURES, WINDOW, HORIZON } from '../models/TemporalGNN';
import { flushTemporal, logRecord, temporalLogOk } from '../utils/temporalLog';
import { levelRank, trafficLevel } from '../utils/trafficColor';

const TICK_MS = 120;
// Saved samples are 5 ticks apart (0.6s). Adjacent ticks are usually identical.
const SAMPLE_EVERY = 5;
const LANE_OFFSET = 8;

const ZONES = [
  { id: 0, x: 150, y: 170, name: 'North-West' },
  { id: 1, x: 450, y: 170, name: 'North-Central' },
  { id: 2, x: 750, y: 170, name: 'North-East' },
  { id: 3, x: 1050, y: 170, name: 'Far-North-East' },
  { id: 4, x: 150, y: 420, name: 'West-Central' },
  { id: 5, x: 450, y: 420, name: 'Central-Hub' },
  { id: 6, x: 750, y: 420, name: 'East-Central' },
  { id: 7, x: 1050, y: 420, name: 'Far-East' },
  { id: 8, x: 150, y: 670, name: 'South-West' },
  { id: 9, x: 450, y: 670, name: 'South-Central' },
  { id: 10, x: 750, y: 670, name: 'South-East' },
  { id: 11, x: 1050, y: 670, name: 'Far-South-East' },
  { id: 12, x: 300, y: 920, name: 'Downtown-West' },
  { id: 13, x: 900, y: 920, name: 'Downtown-East' }
];

// [from, to, capacity, optional midpoint]. Each pair is both directions.
const PAIRS = [
  [0, 1, 12], [1, 2, 12], [2, 3, 12],
  [4, 5, 14], [5, 6, 14], [6, 7, 14],
  [8, 9, 12], [9, 10, 12], [10, 11, 12],
  [12, 13, 10],
  [0, 4, 10], [4, 8, 10],
  [1, 5, 14], [5, 9, 14],
  [2, 6, 12], [6, 10, 12],
  [3, 7, 10], [7, 11, 10],
  [9, 12, 8], [10, 13, 8]
];

const LONG = [
  [0, 6, 6, [450, 290]],
  [3, 9, 5, [750, 420]],
  [8, 13, 6, [525, 800]],
  [2, 11, 5, [900, 420]],
  [4, 13, 7, [525, 670]]
];

// Repeated entries are the corridors that should gather traffic.
const MAIN_PATHS = [
  [0, 1, 2, 3],
  [0, 1, 2, 3],
  [3, 2, 1, 0],
  [4, 5, 6, 7],
  [4, 5, 6, 7],
  [8, 9, 10, 11],
  [1, 5, 9, 12],
  [1, 5, 9, 12],
  [1, 5, 9, 12],
  [2, 6, 10, 13],
  [2, 6, 10, 13],
  [0, 1, 5, 9, 12],
  [0, 1, 5, 9, 12],
  [3, 2, 6, 10, 13],
  [8, 9, 5, 1, 0],
  [4, 5, 9, 10, 11],
  [0, 4, 8],
  [3, 7, 11],
  [11, 10, 9, 8],
  [12, 9, 5, 1],
  [11, 10, 6, 2, 3]
];

const MAIN_PATH_CHANCE = 0.4;

function offsetPolyline(points, offset) {
  const out = new Array(points.length);
  for (let i = 0; i < points.length; i++) {
    let dx;
    let dy;
    if (i === 0) {
      dx = points[1][0] - points[0][0];
      dy = points[1][1] - points[0][1];
    } else if (i === points.length - 1) {
      dx = points[i][0] - points[i - 1][0];
      dy = points[i][1] - points[i - 1][1];
    } else {
      const dx1 = points[i][0] - points[i - 1][0];
      const dy1 = points[i][1] - points[i - 1][1];
      const dx2 = points[i + 1][0] - points[i][0];
      const dy2 = points[i + 1][1] - points[i][1];
      const l1 = Math.hypot(dx1, dy1) || 1;
      const l2 = Math.hypot(dx2, dy2) || 1;
      dx = dx1 / l1 + dx2 / l2;
      dy = dy1 / l1 + dy2 / l2;
    }
    const len = Math.hypot(dx, dy) || 1;
    const nx = dy / len;
    const ny = -dx / len;
    out[i] = [points[i][0] + nx * offset, points[i][1] + ny * offset];
  }
  return out;
}

function polylineLength(points) {
  let len = 0;
  for (let i = 1; i < points.length; i++) {
    len += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
  }
  return len || 1;
}

class Road {
  constructor(id, fromZone, toZone, points, capacity) {
    this.id = id;
    this.fromZone = fromZone;
    this.toZone = toZone;
    this.capacity = capacity;
    this.drawPoints = offsetPolyline(points, LANE_OFFSET);
    this.length = polylineLength(this.drawPoints);
    this.congestionLevel = 0;
    this.vehicleCount = 0;
    this.suggested = 0;
  }

  pointAt(progress) {
    const pts = this.drawPoints;
    const target = Math.max(0, Math.min(0.999, progress)) * this.length;
    let walked = 0;
    for (let i = 1; i < pts.length; i++) {
      const dx = pts[i][0] - pts[i - 1][0];
      const dy = pts[i][1] - pts[i - 1][1];
      const seg = Math.hypot(dx, dy) || 1;
      if (walked + seg >= target || i === pts.length - 1) {
        const t = Math.max(0, Math.min(1, (target - walked) / seg));
        return {
          x: pts[i - 1][0] + dx * t,
          y: pts[i - 1][1] + dy * t,
          angle: Math.atan2(dy, dx)
        };
      }
      walked += seg;
    }
    const last = pts[pts.length - 1];
    return { x: last[0], y: last[1], angle: 0 };
  }
}

function round(value, digits) {
  const p = 10 ** digits;
  return Math.round(value * p) / p;
}

export class Simulation {
  constructor() {
    this.zones = ZONES.map((z) => ({ ...z }));
    this.roads = [];
    this.signals = [];
    this.intersections = [];
    this.vehicles = [];
    this.nextVehicleId = 1;
    this.outgoing = new Map();
    this.signalByRoad = new Map();
    this.roadByPair = new Map();
    this.mode = 'collect';
    this.step = 0;
    this.session = Date.now();
    this.snapshots = [];
    this.learnFrames = [];
    this.pending = new Map();
    this.notices = [];
    this.stats = {
      loss: 1,
      rounds: 0,
      samples: 0,
      mae: 0,
      maeCount: 0,
      cohortTotal: 0,
      cohortDone: 0
    };
    this._buildNetwork();
    this._buildModel();
    this._randomizeLights();
  }

  _addRoad(from, to, capacity, mid) {
    const a = this.zones[from];
    const b = this.zones[to];
    const points = mid
      ? [[a.x, a.y], mid, [b.x, b.y]]
      : [[a.x, a.y], [(a.x + b.x) / 2, (a.y + b.y) / 2], [b.x, b.y]];
    const road = new Road(this.roads.length, from, to, points, capacity);
    this.roads.push(road);
    this.roadByPair.set(`${from}-${to}`, road);
    if (!this.outgoing.has(from)) this.outgoing.set(from, []);
    this.outgoing.get(from).push(road);
    return road;
  }

  _buildNetwork() {
    for (let i = 0; i < PAIRS.length; i++) {
      const [a, b, cap] = PAIRS[i];
      this._addRoad(a, b, cap);
      this._addRoad(b, a, cap);
    }
    for (let i = 0; i < LONG.length; i++) {
      const [a, b, cap, mid] = LONG[i];
      this._addRoad(a, b, cap, mid);
    }

    for (let i = 0; i < this.roads.length; i++) {
      const road = this.roads[i];
      const stop = road.pointAt(0.9);
      const lx = stop.x + Math.sin(stop.angle) * 16;
      const ly = stop.y - Math.cos(stop.angle) * 16;
      const signal = {
        id: this.signals.length,
        roadId: road.id,
        intersectionId: road.toZone,
        fromZone: road.fromZone,
        direction: stop.angle,
        state: 'red',
        timer: 0,
        x: lx,
        y: ly,
        stopX: stop.x,
        stopY: stop.y,
        angle: stop.angle,
        waiting: 0,
        moving: 0,
        avgSpeed: 0,
        outflow: 0,
        departedThisStep: 0,
        congestion: 0,
        prediction: null
      };
      this.signals.push(signal);
      this.signalByRoad.set(road.id, signal);
    }

    const byZone = new Map();
    for (let i = 0; i < this.signals.length; i++) {
      const id = this.signals[i].intersectionId;
      if (!byZone.has(id)) byZone.set(id, []);
      byZone.get(id).push(this.signals[i].id);
    }
    for (let z = 0; z < this.zones.length; z++) {
      const approachIds = byZone.get(z) || [];
      approachIds.sort((a, b) => this.signals[a].direction - this.signals[b].direction);
      this.intersections.push({
        id: z,
        approachIds,
        active: 0,
        phase: 'green',
        timer: 1
      });
    }

    for (let p = 0; p < MAIN_PATHS.length; p++) {
      const path = MAIN_PATHS[p];
      for (let i = 0; i < path.length - 1; i++) {
        if (!this.roadByPair.has(`${path[i]}-${path[i + 1]}`)) {
          throw new Error(`Main path has no road ${path[i]} -> ${path[i + 1]}`);
        }
      }
    }
  }

  _buildModel() {
    const neighbors = this.signals.map(() => []);
    const add = (a, b) => {
      if (a !== b && neighbors[a].indexOf(b) === -1) neighbors[a].push(b);
    };
    for (let i = 0; i < this.intersections.length; i++) {
      const ids = this.intersections[i].approachIds;
      for (let a = 0; a < ids.length; a++) {
        for (let b = 0; b < ids.length; b++) add(ids[a], ids[b]);
      }
    }
    for (let i = 0; i < this.signals.length; i++) {
      const here = this.signals[i].intersectionId;
      const leaving = this.outgoing.get(here) || [];
      for (let k = 0; k < leaving.length; k++) {
        const downstream = this.signalByRoad.get(leaving[k].id);
        add(i, downstream.id);
        add(downstream.id, i);
      }
    }
    this.neighbors = neighbors;
    this.model = new TemporalGNN(this.signals.length, neighbors);
    this.groups = this.intersections.map((inter) => inter.approachIds.slice());
    this.finished = [];
  }

  freshModel() {
    this.model = new TemporalGNN(this.signals.length, this.neighbors);
    this.stats.loss = 1;
    this.stats.rounds = 0;
    this.stats.mae = 0;
    this.stats.maeCount = 0;
    this.learnFrames = [];
  }

  _rand() {
    return this.rng ? this.rng() : Math.random();
  }

  _greenDuration(signalId) {
    return 30 + ((signalId * 17) % 45) + Math.floor(this._rand() * 36);
  }

  _applyPhase(inter) {
    for (let i = 0; i < inter.approachIds.length; i++) {
      const signal = this.signals[inter.approachIds[i]];
      signal.state = i === inter.active ? inter.phase : 'red';
      signal.timer = i === inter.active ? inter.timer : 0;
    }
  }

  _randomizeLights() {
    for (let i = 0; i < this.intersections.length; i++) {
      const inter = this.intersections[i];
      inter.active = Math.floor(this._rand() * inter.approachIds.length);
      inter.phase = 'green';
      const signalId = inter.approachIds[inter.active];
      inter.timer = 8 + Math.floor(this._rand() * this._greenDuration(signalId));
      this._applyPhase(inter);
    }
  }

  tickLights() {
    for (let i = 0; i < this.intersections.length; i++) {
      const inter = this.intersections[i];
      inter.timer -= 1;
      if (inter.timer > 0) {
        const signal = this.signals[inter.approachIds[inter.active]];
        signal.timer = inter.timer;
        continue;
      }
      if (inter.phase === 'green') {
        inter.phase = 'yellow';
        inter.timer = 8;
      } else {
        inter.active = (inter.active + 1) % inter.approachIds.length;
        inter.phase = 'green';
        inter.timer = this._greenDuration(inter.approachIds[inter.active]);
      }
      this._applyPhase(inter);
    }
  }

  // Returns how many intersections currently show more than one green.
  greenConflicts() {
    let conflicts = 0;
    for (let i = 0; i < this.intersections.length; i++) {
      const ids = this.intersections[i].approachIds;
      let greens = 0;
      for (let k = 0; k < ids.length; k++) {
        if (this.signals[ids[k]].state === 'green') greens += 1;
      }
      if (greens > 1) conflicts += 1;
    }
    return conflicts;
  }

  addVehicles(count) {
    if (this.tripDataset && this.tripDataset.vehicles.length > 0) {
      return this._addDatasetVehicles(count);
    }
    const n = count == null ? 4 + Math.floor(Math.random() * 7) : count;
    let main = 0;
    let random = 0;
    for (let i = 0; i < n; i++) {
      const useMain = Math.random() < MAIN_PATH_CHANCE;
      if (useMain) {
        const path = MAIN_PATHS[Math.floor(Math.random() * MAIN_PATHS.length)];
        const start = Math.floor(Math.random() * (path.length - 1));
        const road = this.roadByPair.get(`${path[start]}-${path[start + 1]}`);
        this._place(road, path, start + 1, 'main');
        main += 1;
      } else {
        const road = this.roads[Math.floor(Math.random() * this.roads.length)];
        this._place(road, null, 0, 'random');
        random += 1;
      }
    }
    this.recountRoads();
    return { added: n, main, random, trips: 0 };
  }

  addTripBatch(dataset) {
    const specs = dataset.vehicles;
    const plans = [];
    for (let i = 0; i < specs.length; i++) {
      const spec = specs[i];
      if (plans.length < 4) {
        plans.push(spec.trips.map((trip) => (
          `${this.zones[trip.source].name} → ${this.zones[trip.dest].name}`
        )).join(' · '));
      }
      this.spawnTripVehicle({
        id: this.nextVehicleId++,
        speed: spec.speed,
        trips: spec.trips,
        cohort: true
      }, 'shortest');
    }
    this.stats.cohortTotal += specs.length;
    this.recountRoads();
    return { added: specs.length, main: 0, random: 0, trips: specs.length, plans };
  }

  // After the trip list exists, new cars drive those same chains so training
  // sees the roads the test will use. They stay on the short path while learning.
  _addDatasetVehicles(count) {
    const specs = this.tripDataset.vehicles;
    const n = count == null ? specs.length : count;
    const plans = [];
    for (let i = 0; i < n; i++) {
      const spec = specs[i % specs.length];
      if (plans.length < 4) {
        plans.push(spec.trips.map((trip) => (
          `${this.zones[trip.source].name} → ${this.zones[trip.dest].name}`
        )).join(' · '));
      }
      this.spawnTripVehicle({
        id: this.nextVehicleId++,
        speed: spec.speed,
        trips: spec.trips,
        cohort: true
      }, 'shortest');
    }
    this.stats.cohortTotal += n;
    this.recountRoads();
    return { added: n, main: 0, random: 0, trips: n, plans };
  }

  recountRoads() {
    for (let i = 0; i < this.roads.length; i++) this.roads[i].vehicleCount = 0;
    for (let i = 0; i < this.vehicles.length; i++) {
      const road = this.vehicles[i].road;
      if (road) road.vehicleCount += 1;
    }
    for (let i = 0; i < this.roads.length; i++) {
      const road = this.roads[i];
      road.congestionLevel = Math.min(1, road.vehicleCount / road.capacity);
    }
  }

  _place(road, path, pathCursor, kind) {
    const progress = 0.04 + Math.random() * 0.45;
    const pos = road.pointAt(progress);
    const roll = Math.random();
    const speed = roll < 0.3
      ? 2.2 + Math.random() * 1.0
      : roll < 0.75
        ? 4.4 + Math.random() * 1.4
        : 7.2 + Math.random() * 1.8;
    this.vehicles.push({
      id: this.nextVehicleId++,
      x: pos.x,
      y: pos.y,
      angle: pos.angle,
      speed,
      speedNow: 0,
      road,
      progress,
      path,
      pathCursor,
      kind,
      notice: null,
      stopped: false
    });
  }

  chooseNext(vehicle, zoneId) {
    if (vehicle.trips) return this.chooseTripRoad(vehicle, zoneId);
    const outgoing = this.outgoing.get(zoneId) || [];
    if (outgoing.length === 0) return null;

    let preferred = null;
    let cursor = vehicle.pathCursor;
    if (vehicle.path && vehicle.path[cursor] === zoneId) {
      const nextZone = vehicle.path[cursor + 1];
      if (nextZone != null) {
        preferred = this.roadByPair.get(`${zoneId}-${nextZone}`) || null;
        if (preferred) cursor = cursor + 1;
      }
    }
    if (!preferred) {
      preferred = outgoing[Math.floor(Math.random() * outgoing.length)];
      vehicle.path = null;
    }

    if (this.stats.rounds === 0 || outgoing.length < 2) {
      return { road: preferred, cursor, reroute: false };
    }

    const scoreOf = (road) => this._travelTicks(road, vehicle.speed, true, true);
    let best = outgoing[0];
    let bestScore = scoreOf(best);
    for (let i = 1; i < outgoing.length; i++) {
      const score = scoreOf(outgoing[i]);
      if (score < bestScore) {
        best = outgoing[i];
        bestScore = score;
      }
    }
    if (best !== preferred && bestScore + 12 < scoreOf(preferred)) {
      vehicle.path = null;
      const lines = [
        `Destination: ${this.zones[best.toZone].name}`,
        `Route: ${this.zones[preferred.toZone].name}`,
        `Faster route: ${this.zones[best.toZone].name}`
      ];
      return {
        road: best,
        cursor,
        reroute: true,
        notice: { text: lines.join(' · '), lines }
      };
    }
    return { road: preferred, cursor, reroute: false };
  }

  // Least-cost path. Untrained cost is one hop. Trained cost is the predicted
  // jam on that road plus a hop penalty, so a detour is used only when it
  // avoids real traffic.
  _search(from, to, costOf, avoidFirst) {
    const n = this.zones.length;
    const dist = new Array(n).fill(Infinity);
    const prev = new Array(n).fill(-1);
    const used = new Array(n).fill(false);
    dist[from] = 0;
    for (let k = 0; k < n; k++) {
      let best = -1;
      for (let i = 0; i < n; i++) {
        if (!used[i] && (best < 0 || dist[i] < dist[best])) best = i;
      }
      if (best < 0 || dist[best] === Infinity) break;
      used[best] = true;
      const roads = this.outgoing.get(best) || [];
      for (let r = 0; r < roads.length; r++) {
        const road = roads[r];
        if (best === from && avoidFirst != null && road.toZone === avoidFirst && to !== avoidFirst) {
          continue;
        }
        const next = dist[best] + costOf(road);
        if (next < dist[road.toZone]) {
          dist[road.toZone] = next;
          prev[road.toZone] = best;
        }
      }
    }
    if (dist[to] === Infinity) return null;
    const path = [];
    for (let at = to; at >= 0; at = prev[at]) path.push(at);
    path.reverse();
    return path;
  }

  _searchAvoiding(from, to, costOf, avoidFirst) {
    const path = this._search(from, to, costOf, avoidFirst);
    if (path || avoidFirst == null) return path;
    return this._search(from, to, costOf, null);
  }

  shortestPath(from, to, avoidFirst) {
    if (from === to) return [from];
    return this._searchAvoiding(from, to, () => 1, avoidFirst);
  }

  _loadTargets() {
    return this.signals.map((signal) => Math.min(1, (signal.waiting + signal.moving) / 4));
  }

  _roadsOf(path) {
    const roads = [];
    if (!path) return roads;
    for (let i = 0; i < path.length - 1; i++) {
      const road = this.roadByPair.get(`${path[i]}-${path[i + 1]}`);
      if (road) roads.push(road);
    }
    return roads;
  }

  _releaseSuggested(vehicle) {
    const roads = vehicle.suggestedRoads;
    if (!roads) return;
    for (let i = 0; i < roads.length; i++) {
      roads[i].suggested = Math.max(0, roads[i].suggested - 1);
    }
    vehicle.suggestedRoads = null;
  }

  // Remember that this trained vehicle was sent along the path, including
  // roads it has not reached yet.
  _holdSuggested(vehicle, path) {
    this._releaseSuggested(vehicle);
    const roads = this._roadsOf(path);
    for (let i = 0; i < roads.length; i++) roads[i].suggested += 1;
    vehicle.suggestedRoads = roads;
  }

  _clearSuggested() {
    for (let i = 0; i < this.roads.length; i++) this.roads[i].suggested = 0;
  }

  // Mean green length for this approach. The random part of the real green
  // averages 18 ticks.
  _expectedGreen(signalId) {
    return 30 + ((signalId * 17) % 45) + 18;
  }

  _cycleLength(inter) {
    const ids = inter.approachIds;
    let total = 0;
    for (let i = 0; i < ids.length; i++) total += this._expectedGreen(ids[i]) + 8;
    return Math.max(1, total);
  }

  // Phase of this approach `t` ticks from now. Only the current phase timer
  // is known; later greens use the expected length.
  _phaseAt(signal, t) {
    const inter = this.intersections[signal.intersectionId];
    const ids = inter.approachIds;
    const n = ids.length || 1;
    let idx = inter.active;
    let phase = inter.phase;
    let left = Math.max(1, inter.timer);
    let clock = 0;
    const when = Math.max(0, t);
    for (let guard = 0; guard < 40 && clock + left <= when; guard++) {
      clock += left;
      if (phase === 'green') {
        phase = 'yellow';
        left = 8;
      } else {
        idx = (idx + 1) % n;
        phase = 'green';
        left = this._expectedGreen(ids[idx]);
      }
    }
    return {
      green: phase === 'green' && ids[idx] === signal.id,
      left: Math.max(0, clock + left - when)
    };
  }

  // Ticks from `arriveAt` until this approach is green, and how long that
  // green is expected to last.
  _waitUntilGreen(signal, arriveAt) {
    const now = this._phaseAt(signal, arriveAt);
    if (now.green) return { wait: 0, budget: now.left };
    let t = arriveAt + Math.max(1, now.left);
    for (let k = 0; k < 16; k++) {
      const phase = this._phaseAt(signal, t);
      if (phase.green) return { wait: Math.max(0, t - arriveAt), budget: phase.left };
      t += Math.max(1, phase.left);
    }
    return { wait: 24, budget: this._expectedGreen(signal.id) };
  }

  // Estimated ticks to drive this road and clear its light.
  // soon: the car enters this road now, so the current light phase matters.
  // Later roads use the average wait, because the light will have cycled.
  // Counted together: road length, this car's speed, time until green, whether
  // that green is long enough for the queue, cars already on the road, cars
  // told to use it but not here yet, recent outflow, and the forecast.
  _travelTicks(road, speed, countSuggested, soon) {
    const signal = this.signalByRoad.get(road.id);
    const paceSpeed = Math.max(1, speed || 5);
    const onRoad = road.vehicleCount;
    const sent = countSuggested ? road.suggested || 0 : 0;
    const incoming = Math.max(0, sent - onRoad);
    const forecastCars = !signal || signal.prediction == null ? 0 : signal.prediction * 4;
    const forecastWeight = soon ? 0.7 : 0.35;
    const extraForecast = Math.max(0, forecastCars - onRoad) * forecastWeight;
    // Cars already rolling slow the drive. Cars waiting at the light are a
    // queue at the end of the road, not a crawl along the whole length.
    const movingCars = signal ? signal.moving : onRoad;
    const movingLoad = Math.min(1, (movingCars + extraForecast * 0.5) / road.capacity);
    const pace = Math.max(0.55, 1 - 0.4 * movingLoad);
    const drive = road.length / (paceSpeed * pace);

    let queue = incoming + extraForecast;
    if (signal) {
      queue += signal.waiting + signal.moving * 0.4;
      if (signal.state === 'green' && signal.outflow > 0) {
        queue = Math.max(0, queue - signal.outflow);
      }
    }
    const perCar = Math.max(4, 18 / paceSpeed);
    const need = queue * perCar;
    let wait = need;
    // Delay rises sharply once the cars already on the road, already sent
    // here, and forecast to arrive fill most of the capacity. One more
    // vehicle sent onto that road is what turns it into a jam.
    const committed = onRoad + incoming + extraForecast;
    const fill = committed / Math.max(1, road.capacity);
    if (fill > 0.5) {
      const over = fill - 0.5;
      wait += over * over * 280;
    }
    if (signal && soon) {
      const slot = this._waitUntilGreen(signal, drive);
      wait += slot.wait;
      if (slot.budget + 1 < need) {
        const inter = this.intersections[signal.intersectionId];
        const redGap = this._cycleLength(inter) - this._expectedGreen(signal.id);
        wait += slot.budget + Math.max(0, redGap);
      }
    } else if (signal) {
      const inter = this.intersections[signal.intersectionId];
      const cycle = this._cycleLength(inter);
      const green = Math.min(cycle, this._expectedGreen(signal.id));
      const red = cycle - green;
      wait += (red * red) / (2 * cycle);
    }
    return drive + Math.max(0, wait);
  }

  _pathTicks(path, speed) {
    const roads = this._roadsOf(path);
    let total = 0;
    for (let i = 0; i < roads.length; i++) {
      total += this._travelTicks(roads[i], speed, true, i === 0);
    }
    return total;
  }

  fastestPath(from, to, speed, countSuggested, avoidFirst) {
    if (from === to) return [from];
    return this._searchAvoiding(
      from,
      to,
      (road) => this._travelTicks(road, speed, countSuggested, road.fromZone === from),
      avoidFirst
    );
  }

  // Untrained cars take the fewest intersections. Trained cars take a longer
  // path only when its estimated travel time is clearly lower. The margin
  // grows with each extra intersection, so a small saving does not send the
  // car the long way around.
  // avoidFirst is the intersection the car just left, so the next road cannot
  // turn straight back unless that is the only way to the destination.
  routePath(from, to, guide, speed, avoidFirst) {
    const shortest = this.shortestPath(from, to, avoidFirst);
    if (guide !== 'trained' || !shortest || shortest.length < 2) return shortest;
    const fastest = this.fastestPath(from, to, speed, true, avoidFirst);
    if (!fastest || fastest.length < 2) return shortest;
    if (fastest.length > shortest.length + 3) return shortest;
    if (fastest.length === shortest.length && fastest.join() === shortest.join()) return fastest;
    const fastTime = this._pathTicks(fastest, speed);
    const shortTime = this._pathTicks(shortest, speed);
    const extra = Math.max(0, fastest.length - shortest.length);
    const margin = extra === 0 ? 6 : 10 + extra * 14;
    if (fastTime + margin < shortTime) return fastest;
    return shortest;
  }

  _pathLabel(path) {
    if (!path || path.length < 2) return '';
    const names = [];
    for (let i = 0; i < path.length; i++) names.push(this.zones[path[i]].name);
    return names.join(' → ');
  }

  _routeNotice(destId, route, faster) {
    const lines = [`Destination: ${this.zones[destId].name}`];
    const routeText = this._pathLabel(route);
    if (routeText) lines.push(`Route: ${routeText}`);
    const fasterText = this._pathLabel(faster);
    if (fasterText && fasterText !== routeText) lines.push(`Faster route: ${fasterText}`);
    return { text: lines.join(' · '), lines };
  }

  _scheduleDestFlash(vehicle) {
    vehicle.destFlashAt = this.step + 30 + Math.floor(Math.random() * 140);
  }

  _maybeShowDestination(vehicle) {
    if (!vehicle.trips || vehicle.done) return;
    if (vehicle.notice && this.step <= vehicle.notice.until) return;
    if (vehicle.destFlashAt == null) this._scheduleDestFlash(vehicle);
    if (this.step < vehicle.destFlashAt) return;
    const trip = vehicle.trips[vehicle.tripIndex];
    if (!trip) return;
    const line = `Destination: ${this.zones[trip.dest].name}`;
    vehicle.notice = { text: line, lines: [line], until: this.step + 16 };
    this._scheduleDestFlash(vehicle);
  }

  _showNotice(vehicle, notice) {
    if (!notice) return;
    vehicle.notice = {
      text: notice.text,
      lines: notice.lines,
      until: this.step + 18
    };
    this.notices.push({
      id: vehicle.id,
      text: notice.text,
      lines: notice.lines,
      step: this.step
    });
    if (this.notices.length > 6) this.notices.shift();
  }

  chooseTripRoad(vehicle, zoneId) {
    let trip = vehicle.trips[vehicle.tripIndex];
    if (trip && zoneId === trip.dest) {
      vehicle.tripIndex += 1;
      trip = vehicle.trips[vehicle.tripIndex];
    }
    if (!trip) {
      vehicle.done = true;
      return null;
    }
    const trace = vehicle.trace || [];
    const previous = trace.length >= 2 ? trace[trace.length - 2] : null;
    const avoid = previous != null && previous !== trip.dest ? previous : null;
    const shortest = this.shortestPath(zoneId, trip.dest, avoid);
    const guided = vehicle.cohort && this.stats.rounds > 0 ? 'trained' : vehicle.guide;
    const chosen = this.routePath(zoneId, trip.dest, guided, vehicle.speed, avoid) || shortest;
    if (!chosen || chosen.length < 2) {
      vehicle.done = true;
      return null;
    }
    const road = this.roadByPair.get(`${chosen[0]}-${chosen[1]}`);
    if (!road) {
      vehicle.done = true;
      return null;
    }
    const reroute = guided === 'trained' && !!shortest && shortest[1] !== chosen[1];
    const notice = reroute ? this._routeNotice(trip.dest, shortest, chosen) : null;
    return { road, cursor: 0, path: chosen, reroute, notice, hold: guided === 'trained' };
  }

  arrive(vehicle) {
    const signal = this.signalByRoad.get(vehicle.road.id);
    signal.departedThisStep += 1;
    const zoneId = vehicle.road.toZone;
    if (vehicle.trace) vehicle.trace.push(zoneId);
    this._releaseSuggested(vehicle);
    const choice = this.chooseNext(vehicle, zoneId);
    if (!choice || vehicle.done) {
      vehicle.done = true;
      vehicle.road = null;
      this._archive(vehicle);
      return;
    }
    if (choice.hold && choice.path) this._holdSuggested(vehicle, choice.path);
    vehicle.road = choice.road;
    if (choice.path) vehicle.path = choice.path;
    vehicle.progress = 0.02;
    vehicle.pathCursor = choice.cursor;
    vehicle.stopped = false;
    if (choice.notice) this._showNotice(vehicle, choice.notice);
    else vehicle.notice = null;
    if (choice.reroute) vehicle.kind = 'rerouted';
    const pos = vehicle.road.pointAt(vehicle.progress);
    vehicle.x = pos.x;
    vehicle.y = pos.y;
    vehicle.angle = pos.angle;
  }

  tickVehicles() {
    const grouped = new Map();
    for (let i = 0; i < this.vehicles.length; i++) {
      const vehicle = this.vehicles[i];
      if (!vehicle.road || vehicle.done) continue;
      if (vehicle.notice && this.step > vehicle.notice.until) vehicle.notice = null;
      this._maybeShowDestination(vehicle);
      const list = grouped.get(vehicle.road.id);
      if (list) list.push(vehicle);
      else grouped.set(vehicle.road.id, [vehicle]);
    }

    grouped.forEach((list, roadId) => {
      list.sort((a, b) => b.progress - a.progress);
      const road = this.roads[roadId];
      const open = this.signalByRoad.get(road.id).state === 'green';
      const hold = Math.min(0.12, 28 / road.length);
      const follow = Math.min(0.08, 22 / road.length);
      let front = 1.08;
      for (let i = 0; i < list.length; i++) {
        const vehicle = list[i];
        const limit = Math.min(open ? 1 : 1 - hold, front - follow);
        const pace = i === 0 && open ? 1 : (1 - 0.65 * road.congestionLevel);
        const px = vehicle.speed * pace;
        let next = vehicle.progress + px / road.length;
        if (next >= 0.995 && open && limit >= 0.995) {
          this.arrive(vehicle);
          continue;
        }
        if (next > limit) {
          next = Math.max(0, limit);
          vehicle.stopped = true;
          vehicle.speedNow = 0;
        } else {
          vehicle.stopped = false;
          vehicle.speedNow = px;
        }
        vehicle.progress = next;
        const pos = road.pointAt(next);
        vehicle.x = pos.x;
        vehicle.y = pos.y;
        vehicle.angle = pos.angle;
        front = vehicle.progress;
      }
    });
  }

  measure() {
    for (let i = 0; i < this.signals.length; i++) {
      const signal = this.signals[i];
      signal.waiting = 0;
      signal.moving = 0;
      signal.avgSpeed = 0;
      signal.outflow = signal.departedThisStep;
      signal.departedThisStep = 0;
    }
    for (let i = 0; i < this.vehicles.length; i++) {
      const vehicle = this.vehicles[i];
      if (!vehicle.road || vehicle.done) continue;
      const signal = this.signalByRoad.get(vehicle.road.id);
      if (vehicle.stopped && vehicle.progress > 0.72) signal.waiting += 1;
      else signal.moving += 1;
      signal.avgSpeed += vehicle.speedNow;
    }
    for (let i = 0; i < this.signals.length; i++) {
      const signal = this.signals[i];
      const total = signal.waiting + signal.moving;
      if (total > 0) signal.avgSpeed /= total;
      const road = this.roads[signal.roadId];
      road.vehicleCount = total;
      road.congestionLevel = Math.min(1, total / road.capacity);
      signal.congestion = road.congestionLevel;
    }
  }

  features() {
    const rows = new Array(this.signals.length);
    for (let i = 0; i < this.signals.length; i++) {
      const s = this.signals[i];
      const road = this.roads[s.roadId];
      const incoming = Math.max(0, (road.suggested || 0) - road.vehicleCount);
      rows[i] = [
        s.waiting / 8,
        s.moving / 8,
        s.avgSpeed / 8,
        Math.sin(s.direction),
        Math.cos(s.direction),
        s.state === 'green' ? 1 : 0,
        s.state === 'yellow' ? 1 : 0,
        s.timer / 100,
        s.outflow / 4,
        Math.min(1, incoming / 4)
      ];
    }
    return rows;
  }

  _remember() {
    const frame = this.snapshots[this.snapshots.length - 1];
    this.learnFrames.push({
      features: frame,
      targets: this._loadTargets()
    });
    if (this.learnFrames.length > 3000) this.learnFrames.shift();
  }

  // Replay the finished trips. Empty stretches are skipped so they do not
  // wipe out the congestion the cars actually created.
  fitLearnedTrips() {
    const frames = this.learnFrames;
    if (!frames || frames.length < WINDOW + HORIZON) return 0;
    let added = 0;
    let loss = this.stats.loss;
    for (let epoch = 0; epoch < 2; epoch++) {
      for (let i = WINDOW - 1; i < frames.length - HORIZON; i += 8) {
        const targets = frames[i + HORIZON].targets;
        let peak = 0;
        for (let k = 0; k < targets.length; k++) if (targets[k] > peak) peak = targets[k];
        if (peak < 0.25) continue;
        const window = [];
        for (let t = i - (WINDOW - 1); t <= i; t++) window.push(frames[t].features);
        const next = this.model.federatedRound(window, targets, this.groups, 0.03);
        if (Number.isFinite(next)) {
          loss = next;
          added += 1;
        }
      }
    }
    this.stats.loss = loss;
    this.stats.rounds += added;
    return added;
  }

  _train() {
    let active = 0;
    for (let i = 0; i < this.vehicles.length; i++) {
      if (this.vehicles[i].road && !this.vehicles[i].done) active += 1;
    }
    if (active === 0) return;
    if (this.snapshots.length < WINDOW + HORIZON) return;
    if (this.step % SAMPLE_EVERY !== 0) return;
    const end = this.snapshots.length - HORIZON;
    const window = this.snapshots.slice(end - WINDOW, end);
    const targets = this._loadTargets();
    let peak = 0;
    for (let i = 0; i < targets.length; i++) if (targets[i] > peak) peak = targets[i];
    if (peak < 0.25) return;
    const loss = this.model.federatedRound(window, targets, this.groups, 0.03);
    if (Number.isFinite(loss)) {
      this.stats.loss = loss;
      this.stats.rounds += 1;
    }
  }

  _predict() {
    if (this.pending.has(this.step)) {
      const forecast = this.pending.get(this.step);
      this.pending.delete(this.step);
      if (this.stats.rounds > 0 && this.step % SAMPLE_EVERY === 0) {
        const targets = this._loadTargets();
        const lights = new Array(this.signals.length);
        for (let i = 0; i < this.signals.length; i++) {
          const actual = targets[i];
          const predicted = forecast[i];
          const err = Math.abs(predicted - actual);
          this.stats.maeCount += 1;
          this.stats.mae += (err - this.stats.mae) / this.stats.maeCount;
          lights[i] = {
            id: i,
            intersection: this.signals[i].intersectionId,
            predicted: round(predicted, 3),
            actual: round(actual, 3)
          };
        }
        logRecord({
          type: 'prediction',
          session: this.session,
          mode: this.mode,
          step: this.step,
          horizon: HORIZON,
          lights
        });
      }
    }

    if (this.snapshots.length < WINDOW) return;
    const window = this.snapshots.slice(-WINDOW);
    const { y } = this.model.forward(window, false);
    for (let i = 0; i < this.signals.length; i++) {
      this.signals[i].prediction = y[i];
    }
    this.pending.set(this.step + HORIZON, y.slice());
  }

  _logSignals() {
    if (this.step % SAMPLE_EVERY !== 0) return;
    const lights = new Array(this.signals.length);
    for (let i = 0; i < this.signals.length; i++) {
      const s = this.signals[i];
      lights[i] = {
        id: s.id,
        intersection: s.intersectionId,
        from: s.fromZone,
        dir: round(s.direction, 3),
        waiting: s.waiting,
        moving: s.moving,
        speed: round(s.avgSpeed, 2),
        state: s.state,
        phaseLeft: s.timer,
        outflow: s.outflow,
        congestion: round(s.congestion, 3),
        suggested: Math.max(0, (this.roads[s.roadId].suggested || 0) - this.roads[s.roadId].vehicleCount)
      };
    }
    logRecord({
      type: 'signals',
      session: this.session,
      mode: this.mode,
      step: this.step,
      lights
    });
    this.stats.samples += 1;
  }

  advance() {
    this.step += 1;
    this.tickLights();
    this.tickVehicles();
    this.vehicles = this.vehicles.filter((vehicle) => !vehicle.done);
    this.measure();
    const frame = this.features();
    if (frame[0].length !== FEATURES) {
      throw new Error('Feature width does not match the temporal model');
    }
    this.snapshots.push(frame);
    if (this.snapshots.length > WINDOW + HORIZON + 2) {
      this.snapshots.shift();
    }
    if (this.mode === 'collect') {
      this._remember();
      this._train();
    }
    this._predict();
    this._logSignals();
    flushTemporal();
  }

  resetEnvironment() {
    this.vehicles = [];
    this.step = 0;
    this.snapshots = [];
    this.pending.clear();
    this.notices = [];
    this.stats.mae = 0;
    this.stats.maeCount = 0;
    this.stats.cohortTotal = 0;
    this.stats.cohortDone = 0;
    this.learnFrames = [];
    for (let i = 0; i < this.signals.length; i++) {
      const signal = this.signals[i];
      signal.waiting = 0;
      signal.moving = 0;
      signal.avgSpeed = 0;
      signal.outflow = 0;
      signal.departedThisStep = 0;
      signal.congestion = 0;
      signal.prediction = null;
    }
    for (let i = 0; i < this.roads.length; i++) {
      this.roads[i].congestionLevel = 0;
      this.roads[i].vehicleCount = 0;
    }
    this._clearSuggested();
    this._randomizeLights();
  }

  setMode(mode) {
    this.mode = mode;
  }

  _archive(vehicle) {
    if (vehicle.archived) return;
    vehicle.archived = true;
    if (vehicle.cohort) this.stats.cohortDone += 1;
    this.finished.push({
      id: vehicle.id,
      speed: Math.round(vehicle.speed * 10) / 10,
      trips: vehicle.trips,
      trace: vehicle.trace ? vehicle.trace.slice() : []
    });
  }

  resetForExperiment() {
    this.vehicles = [];
    this.finished = [];
    this.step = 0;
    this.snapshots = [];
    this.pending.clear();
    this.notices = [];
    for (let i = 0; i < this.signals.length; i++) {
      const signal = this.signals[i];
      signal.waiting = 0;
      signal.moving = 0;
      signal.avgSpeed = 0;
      signal.outflow = 0;
      signal.departedThisStep = 0;
      signal.congestion = 0;
      signal.prediction = null;
    }
    for (let i = 0; i < this.roads.length; i++) {
      this.roads[i].congestionLevel = 0;
      this.roads[i].vehicleCount = 0;
    }
    this._clearSuggested();
    this._randomizeLights();
  }

  spawnTripVehicle(spec, guide) {
    const trip = spec.trips[0];
    const guided = spec.cohort && this.stats.rounds > 0 ? 'trained' : guide;
    const shortest = this.shortestPath(trip.source, trip.dest);
    const path = this.routePath(trip.source, trip.dest, guided, spec.speed) || shortest;
    const road = this.roadByPair.get(`${path[0]}-${path[1]}`);
    const progress = 0.04;
    const pos = road.pointAt(progress);
    const rerouted = guided === 'trained' && shortest && shortest[1] !== path[1];
    const vehicle = {
      id: spec.id,
      x: pos.x,
      y: pos.y,
      angle: pos.angle,
      speed: spec.speed,
      speedNow: 0,
      road,
      progress,
      path,
      pathCursor: 0,
      kind: rerouted ? 'rerouted' : 'main',
      notice: null,
      stopped: false,
      trips: spec.trips,
      tripIndex: 0,
      guide: guided,
      cohort: !!spec.cohort,
      trace: [trip.source],
      done: false,
      suggestedRoads: null,
      destFlashAt: null
    };
    this.vehicles.push(vehicle);
    if (guided === 'trained') this._holdSuggested(vehicle, path);
    const changed = guided === 'trained' && shortest && path && shortest.join() !== path.join();
    if (changed) this._showNotice(vehicle, this._routeNotice(trip.dest, shortest, path));
    this._scheduleDestFlash(vehicle);
  }

  warmPredictions(frames) {
    const start = Math.max(0, frames.length - WINDOW);
    this.snapshots = [];
    for (let i = start; i < frames.length; i++) this.snapshots.push(frames[i].features);
    if (this.snapshots.length < WINDOW) return;
    const { y } = this.model.forward(this.snapshots.slice(-WINDOW), false);
    for (let i = 0; i < this.signals.length; i++) this.signals[i].prediction = y[i];
  }

  experimentAdvance(guide) {
    this.step += 1;
    this.tickLights();
    this.tickVehicles();
    this.vehicles = this.vehicles.filter((vehicle) => !vehicle.done);
    this.measure();
    let load = 0;
    let peak = 0;
    let waiting = 0;
    for (let i = 0; i < this.roads.length; i++) {
      const level = this.roads[i].congestionLevel;
      load += level;
      if (level > peak) peak = level;
    }
    load /= this.roads.length;
    for (let i = 0; i < this.signals.length; i++) waiting += this.signals[i].waiting;
    const features = this.features();
    const targets = this.signals.map((signal) => signal.congestion);
    if (guide === 'trained') {
      this.snapshots.push(features);
      if (this.snapshots.length > WINDOW) this.snapshots.shift();
      if (this.snapshots.length >= WINDOW) {
        const { y } = this.model.forward(this.snapshots, false);
        for (let i = 0; i < this.signals.length; i++) this.signals[i].prediction = y[i];
      }
    }
    return { load, peak, waiting, features, targets };
  }

  panel() {
    const ranked = this.signals.map((s) => {
      const road = this.roads[s.roadId];
      const vehicles = road.vehicleCount;
      const level = trafficLevel(vehicles, road.congestionLevel);
      return {
        id: s.id,
        name: `${this.zones[s.intersectionId].name}`,
        from: this.zones[s.fromZone].name,
        state: s.state,
        waiting: s.waiting,
        moving: s.moving,
        vehicles,
        speed: s.avgSpeed,
        load: road.congestionLevel,
        level: level.id,
        levelLabel: level.label,
        color: level.color,
        tint: level.tint,
        prediction: s.prediction,
        phaseLeft: s.timer
      };
    });
    ranked.sort((a, b) => (
      levelRank(b.level) - levelRank(a.level)
      || b.load - a.load
      || b.vehicles - a.vehicles
    ));
    let greens = 0;
    let yellows = 0;
    for (let i = 0; i < this.intersections.length; i++) {
      if (this.intersections[i].phase === 'green') greens += 1;
      else yellows += 1;
    }
    return {
      mode: this.mode,
      step: this.step,
      vehicles: this.vehicles.length,
      cohortTotal: this.stats.cohortTotal,
      cohortDone: this.stats.cohortDone,
      samples: this.stats.samples,
      loss: this.stats.loss,
      rounds: this.stats.rounds,
      accuracy: this.stats.maeCount > 0 ? Math.max(0, (1 - this.stats.mae) * 100) : null,
      saving: temporalLogOk(),
      greens,
      yellows,
      conflicts: this.greenConflicts(),
      busiest: ranked.slice(0, 6),
      notices: this.notices.slice()
    };
  }
}

export function createSimulation() {
  return new Simulation();
}

export { TICK_MS };
