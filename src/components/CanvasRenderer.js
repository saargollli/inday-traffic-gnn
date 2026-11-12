import { useEffect, useRef } from 'react';
import { TRAFFIC_LEVELS, trafficLevel } from '../utils/trafficColor';

function rounded(ctx, x, y, w, h, r) {
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
  else ctx.rect(x, y, w, h);
}

function drawRoad(ctx, road, level) {
  const pts = road.drawPoints;
  ctx.strokeStyle = level.id === 'empty' ? '#e4ebf2' : '#d5dde8';
  ctx.lineWidth = 14;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.stroke();

  ctx.strokeStyle = level.color;
  ctx.lineWidth = level.id === 'empty' ? 4 : 7;
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.stroke();

  const mid = road.pointAt(0.55);
  ctx.save();
  ctx.translate(mid.x, mid.y);
  ctx.rotate(mid.angle);
  ctx.fillStyle = level.id === 'empty' ? '#94a3b8' : '#ffffff';
  ctx.beginPath();
  ctx.moveTo(7, 0);
  ctx.lineTo(-5, 4);
  ctx.lineTo(-5, -4);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawSignal(ctx, signal) {
  const x = signal.x;
  const y = signal.y;
  ctx.fillStyle = '#1f2937';
  rounded(ctx, x - 7, y - 16, 14, 32, 3);
  ctx.fill();

  const lamps = [
    ['red', -10],
    ['yellow', 0],
    ['green', 10]
  ];
  for (let i = 0; i < lamps.length; i++) {
    const on = signal.state === lamps[i][0];
    ctx.beginPath();
    ctx.arc(x, y + lamps[i][1], 3.5, 0, Math.PI * 2);
    if (!on) ctx.fillStyle = '#4b5563';
    else if (lamps[i][0] === 'red') ctx.fillStyle = '#ef4444';
    else if (lamps[i][0] === 'yellow') ctx.fillStyle = '#facc15';
    else ctx.fillStyle = '#22c55e';
    ctx.fill();
  }

  if (signal.state === 'green') {
    const seconds = Math.max(1, Math.round(signal.timer * 0.12));
    ctx.fillStyle = '#14532d';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`${seconds}s`, x + 10, y + 4);
  }
  if (signal.waiting > 0) {
    ctx.fillStyle = '#7f1d1d';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(String(signal.waiting), x, y - 22);
  }
}

function drawVehicle(ctx, vehicle) {
  ctx.save();
  ctx.translate(vehicle.x, vehicle.y);
  ctx.rotate(vehicle.angle);
  let fill = '#64748b';
  if (vehicle.kind === 'main') fill = '#1d4ed8';
  if (vehicle.kind === 'rerouted') fill = '#ea580c';
  const length = vehicle.speed < 3.6 ? 11 : vehicle.speed > 6.8 ? 18 : 14;
  ctx.fillStyle = fill;
  rounded(ctx, -length / 2, -4, length, 8, 2);
  ctx.fill();
  ctx.fillStyle = '#f8fafc';
  ctx.fillRect(length / 2 - 6, -2.5, 3, 5);
  ctx.restore();

  if (!vehicle.notice && vehicle.speedNow > 0.4) {
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#334155';
    ctx.fillText(String(Math.round(vehicle.speed)), vehicle.x, vehicle.y - 10);
  }

  if (vehicle.notice) {
    const lines = vehicle.notice.lines && vehicle.notice.lines.length
      ? vehicle.notice.lines
      : [vehicle.notice.text];
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'center';
    let width = 0;
    for (let i = 0; i < lines.length; i++) {
      width = Math.max(width, ctx.measureText(lines[i]).width);
    }
    const lineH = 13;
    const height = lines.length * lineH + 8;
    let top = vehicle.y - 16 - height;
    if (top < 6) top = vehicle.y + 14;
    ctx.fillStyle = '#fff7ed';
    ctx.strokeStyle = '#fdba74';
    ctx.lineWidth = 1;
    rounded(ctx, vehicle.x - width / 2 - 6, top, width + 12, height, 4);
    ctx.fill();
    ctx.stroke();
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].indexOf('Faster route:') === 0) ctx.fillStyle = '#c2410c';
      else if (lines[i].indexOf('Destination:') === 0) ctx.fillStyle = '#1e3a8a';
      else ctx.fillStyle = '#334155';
      ctx.fillText(lines[i], vehicle.x, top + 12 + i * lineH);
    }
  }
}

function pointToRect(px, py, rect) {
  const dx = Math.max(rect.x - px, 0, px - (rect.x + rect.w));
  const dy = Math.max(rect.y - py, 0, py - (rect.y + rect.h));
  return Math.hypot(dx, dy);
}

function roadClearance(rect, roads) {
  let nearest = Infinity;
  for (let r = 0; r < roads.length; r++) {
    const pts = roads[r].drawPoints;
    for (let i = 1; i < pts.length; i++) {
      for (let s = 0; s <= 6; s++) {
        const t = s / 6;
        const px = pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * t;
        const py = pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * t;
        nearest = Math.min(nearest, pointToRect(px, py, rect));
      }
    }
  }
  return nearest;
}

function boxesOverlap(a, b) {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

let labelSpots = null;

function placeLabels(ctx, sim) {
  ctx.font = 'bold 11px sans-serif';
  const placed = [];
  const zones = sim.zones.slice().sort((a, b) => {
    const degree = (zone) => (sim.outgoing.get(zone.id) || []).length;
    return degree(b) - degree(a);
  });
  const spots = new Array(sim.zones.length);
  for (let z = 0; z < zones.length; z++) {
    const zone = zones[z];
    const width = ctx.measureText(zone.name).width;
    const height = 14;
    let best = null;
    let bestScore = -Infinity;
    for (let k = 0; k < 16; k++) {
      const angle = (Math.PI * 2 * k) / 16 - Math.PI / 2;
      const ux = Math.cos(angle);
      const uy = Math.sin(angle);
      const base = 34 + Math.abs(ux) * width / 2 + Math.abs(uy) * height / 2;
      for (let step = 0; step < 2; step++) {
        const reach = base + step * 16;
        const cx = zone.x + ux * reach;
        const cy = zone.y + uy * reach;
        const rect = { x: cx - width / 2 - 4, y: cy - height / 2 - 2, w: width + 8, h: height + 4 };
        if (rect.x < 6 || rect.y < 48 || rect.x + rect.w > 1194 || rect.y + rect.h > 1072) continue;
        let blocked = false;
        for (let p = 0; p < placed.length; p++) {
          if (boxesOverlap(rect, placed[p])) blocked = true;
        }
        if (blocked) continue;
        const clearance = roadClearance(rect, sim.roads);
        if (clearance > bestScore) {
          bestScore = clearance;
          best = { x: cx, y: cy, rect };
        }
      }
    }
    if (!best) best = { x: zone.x, y: zone.y - 36, rect: null };
    spots[zone.id] = best;
    if (best.rect) placed.push(best.rect);
  }
  return spots;
}

function paint(ctx, sim) {
  ctx.clearRect(0, 0, 1200, 1080);
  ctx.fillStyle = '#f5f8fc';
  ctx.fillRect(0, 0, 1200, 1080);

  ctx.strokeStyle = '#e6edf5';
  ctx.lineWidth = 1;
  for (let x = 0; x <= 1200; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, 1080);
    ctx.stroke();
  }
  for (let y = 0; y <= 1080; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(1200, y);
    ctx.stroke();
  }

  for (let i = 0; i < sim.roads.length; i++) {
    const road = sim.roads[i];
    drawRoad(ctx, road, trafficLevel(road.vehicleCount, road.congestionLevel));
  }

  for (let i = 0; i < sim.zones.length; i++) {
    const zone = sim.zones[i];
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(zone.x, zone.y, 22, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  if (!labelSpots) labelSpots = placeLabels(ctx, sim);
  ctx.font = 'bold 11px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < sim.zones.length; i++) {
    const zone = sim.zones[i];
    const spot = labelSpots[zone.id];
    const width = ctx.measureText(zone.name).width;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.94)';
    rounded(ctx, spot.x - width / 2 - 4, spot.y - 8, width + 8, 16, 4);
    ctx.fill();
    ctx.fillStyle = '#0f172a';
    ctx.fillText(zone.name, spot.x, spot.y);
  }
  ctx.textBaseline = 'alphabetic';

  for (let i = 0; i < sim.signals.length; i++) drawSignal(ctx, sim.signals[i]);
  for (let i = 0; i < sim.vehicles.length; i++) drawVehicle(ctx, sim.vehicles[i]);
  drawLegend(ctx);
}

function drawLegend(ctx) {
  const x = 16;
  const y = 14;
  const width = 430;
  ctx.fillStyle = 'rgba(255, 255, 255, 0.94)';
  ctx.strokeStyle = '#d5dfeb';
  ctx.lineWidth = 1;
  rounded(ctx, x, y, width, 28, 8);
  ctx.fill();
  ctx.stroke();
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  let cursor = x + 10;
  for (let i = 0; i < TRAFFIC_LEVELS.length; i++) {
    const level = TRAFFIC_LEVELS[i];
    ctx.fillStyle = level.color;
    rounded(ctx, cursor, y + 9, 18, 10, 3);
    ctx.fill();
    cursor += 24;
    ctx.fillStyle = '#1e293b';
    ctx.fillText(level.label, cursor, y + 14);
    cursor += ctx.measureText(level.label).width + 14;
  }
  ctx.textBaseline = 'alphabetic';
}

export default function CanvasRenderer({ simRef }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    let ctx = null;
    try {
      ctx = canvas.getContext('2d');
    } catch (err) {
      ctx = null;
    }
    if (!ctx) return undefined;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = 1200 * ratio;
    canvas.height = 1080 * ratio;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

    let frame = 0;
    const loop = () => {
      if (simRef.current) paint(ctx, simRef.current);
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [simRef]);

  return (
    <div className="net-stage">
      <canvas
        ref={canvasRef}
        className="net-canvas"
        aria-label="Road network with a signal on every incoming road"
      />
    </div>
  );
}
