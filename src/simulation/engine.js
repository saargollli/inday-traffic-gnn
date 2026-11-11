import { TemporalGNN, FEATURES, WINDOW, HORIZON } from '../models/TemporalGNN';
import { flushTemporal, logRecord, temporalLogOk } from '../utils/temporalLog';
import { levelRank, trafficLevel } from '../utils/trafficColor';

const TICK_MS = 120;
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
const REROUTE_GAP = 0.12;

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
    this.pending = new Map();
    this.notices = [];
    this.stats = {
      loss: 1,
      rounds: 0,
      samples: 0,
      mae: 0,
      maeCount: 0
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
    return { added: n, main, random };
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

    const ready = this.mode === 'test' && this.stats.rounds > 0 && this.snapshots.length >= WINDOW;
    if (!ready || outgoing.length < 2) {
      return { road: preferred, cursor, reroute: false };
    }
    for (let i = 0; i < outgoing.length; i++) {
      if (this.signalByRoad.get(outgoing[i].id).prediction == null) {
        return { road: preferred, cursor, reroute: false };
      }
    }

    const scoreOf = (road) => {
      const signal = this.signalByRoad.get(road.id);
      return signal.prediction == null ? signal.congestion : signal.prediction;
    };
    let best = outgoing[0];
    let bestScore = scoreOf(best);
    for (let i = 1; i < outgoing.length; i++) {
      const score = scoreOf(outgoing[i]);
      if (score < bestScore) {
        best = outgoing[i];
        bestScore = score;
      }
    }
    if (best !== preferred && bestScore + REROUTE_GAP < scoreOf(preferred)) {
      vehicle.path = null;
      const dest = this.zones[best.toZone].name;
      return {
        road: best,
        cursor,
        reroute: true,
        notice: `Other way: ${dest}`
      };
    }
    return { road: preferred, cursor, reroute: false };
  }

  // Least-cost path. Untrained cost is one hop. Trained cost is the predicted
  // jam on that road plus a hop penalty, so a detour is used only when it
  // avoids real traffic.
  _search(from, to, costOf) {
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

  shortestPath(from, to) {
    if (from === to) return [from];
    return this._search(from, to, () => 1);
  }

  lightestPath(from, to) {
    if (from === to) return [from];
    return this._search(from, to, (road) => {
      const signal = this.signalByRoad.get(road.id);
      const jam = signal.prediction == null ? signal.congestion : signal.prediction;
      return 0.35 + jam;
    });
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
    const shortest = this.shortestPath(zoneId, trip.dest);
    const chosen = vehicle.guide === 'trained'
      ? (this.lightestPath(zoneId, trip.dest) || shortest)
      : shortest;
    if (!chosen || chosen.length < 2) {
      vehicle.done = true;
      return null;
    }
    const road = this.roadByPair.get(`${chosen[0]}-${chosen[1]}`);
    if (!road) {
      vehicle.done = true;
      return null;
    }
    const reroute = vehicle.guide === 'trained' && !!shortest && shortest[1] !== chosen[1];
    let notice = null;
    if (vehicle.guide === 'trained') {
      const names = [];
      for (let i = 1; i < chosen.length && names.length < 3; i++) names.push(this.zones[chosen[i]].name);
      notice = `${reroute ? 'Lighter path' : 'Path'}: ${names.join(' → ')}`;
    }
    return { road, cursor: 0, path: chosen, reroute, notice };
  }

  arrive(vehicle) {
    const signal = this.signalByRoad.get(vehicle.road.id);
    signal.departedThisStep += 1;
    const zoneId = vehicle.road.toZone;
    if (vehicle.trace) vehicle.trace.push(zoneId);
    const choice = this.chooseNext(vehicle, zoneId);
    if (!choice || vehicle.done) {
      vehicle.done = true;
      vehicle.road = null;
      this._archive(vehicle);
      return;
    }
    vehicle.road = choice.road;
    if (choice.path) vehicle.path = choice.path;
    vehicle.progress = 0.02;
    vehicle.pathCursor = choice.cursor;
    vehicle.stopped = false;
    if (choice.notice) {
      vehicle.notice = { text: choice.notice, until: this.step + 40 };
    }
    if (choice.reroute) {
      vehicle.notice = { text: choice.notice, until: this.step + 45 };
      vehicle.kind = 'rerouted';
      this.notices.push({ id: vehicle.id, text: choice.notice, step: this.step });
      if (this.notices.length > 8) this.notices.shift();
    }
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
      const list = grouped.get(vehicle.road.id);
      if (list) list.push(vehicle);
      else grouped.set(vehicle.road.id, [vehicle]);
    }

    grouped.forEach((list, roadId) => {
      list.sort((a, b) => b.progress - a.progress);
      const road = this.roads[roadId];
      const open = this.signalByRoad.get(road.id).state === 'green';
      let front = 1.08;
      for (let i = 0; i < list.length; i++) {
        const vehicle = list[i];
        const limit = Math.min(open ? 1 : 0.9, front - 0.05);
        const px = vehicle.speed * (1 - 0.65 * road.congestionLevel);
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
      const signal = this.signalByRoad.get(vehicle.road.id);
      if (vehicle.stopped && vehicle.progress > 0.72) signal.waiting += 1;
      else signal.moving += 1;
      signal.avgSpeed += vehicle.speedNow;
    }
    for (let i = 0; i < this.signals.length; i++) {
      const signal = this.signals[i];
      const total = signal.waiting + signal.moving;
      if (total > 0) signal.avgSpeed /= total;
      signal.congestion = Math.min(1, signal.waiting / 6);
      const road = this.roads[signal.roadId];
      road.vehicleCount = total;
      road.congestionLevel = Math.min(1, total / road.capacity);
    }
  }

  features() {
    const rows = new Array(this.signals.length);
    for (let i = 0; i < this.signals.length; i++) {
      const s = this.signals[i];
      rows[i] = [
        s.waiting / 8,
        s.moving / 8,
        s.avgSpeed / 8,
        Math.sin(s.direction),
        Math.cos(s.direction),
        s.state === 'green' ? 1 : 0,
        s.state === 'yellow' ? 1 : 0,
        s.timer / 100,
        s.outflow / 4
      ];
    }
    return rows;
  }

  _train() {
    if (this.vehicles.length === 0) return;
    if (this.snapshots.length < WINDOW + HORIZON) return;
    if (this.step % 5 !== 0) return;
    const end = this.snapshots.length - HORIZON;
    const window = this.snapshots.slice(end - WINDOW, end);
    const targets = this.signals.map((s) => s.congestion);
    const loss = this.model.federatedRound(window, targets, this.groups, 0.05);
    if (Number.isFinite(loss)) {
      this.stats.loss = loss;
      this.stats.rounds += 1;
    }
  }

  _predict() {
    if (this.pending.has(this.step)) {
      const forecast = this.pending.get(this.step);
      this.pending.delete(this.step);
      if (this.stats.rounds > 0) {
        const lights = new Array(this.signals.length);
        for (let i = 0; i < this.signals.length; i++) {
          const actual = this.signals[i].congestion;
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
        congestion: round(s.congestion, 3)
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
    this.measure();
    const frame = this.features();
    if (frame[0].length !== FEATURES) {
      throw new Error('Feature width does not match the temporal model');
    }
    this.snapshots.push(frame);
    if (this.snapshots.length > WINDOW + HORIZON + 2) {
      this.snapshots.shift();
    }
    if (this.mode === 'collect') this._train();
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
    this._randomizeLights();
  }

  setMode(mode) {
    this.mode = mode;
  }

  _archive(vehicle) {
    if (vehicle.archived) return;
    vehicle.archived = true;
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
    this._randomizeLights();
  }

  spawnTripVehicle(spec, guide) {
    const trip = spec.trips[0];
    const path = guide === 'trained'
      ? (this.lightestPath(trip.source, trip.dest) || this.shortestPath(trip.source, trip.dest))
      : this.shortestPath(trip.source, trip.dest);
    const road = this.roadByPair.get(`${path[0]}-${path[1]}`);
    const progress = 0.04;
    const pos = road.pointAt(progress);
    this.vehicles.push({
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
      kind: 'main',
      notice: null,
      stopped: false,
      trips: spec.trips,
      tripIndex: 0,
      guide,
      trace: [trip.source],
      done: false
    });
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
