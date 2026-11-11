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
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    const width = ctx.measureText(vehicle.notice.text).width;
    ctx.fillStyle = '#fff7ed';
    ctx.strokeStyle = '#fdba74';
    ctx.lineWidth = 1;
    rounded(ctx, vehicle.x - width / 2 - 6, vehicle.y - 28, width + 12, 16, 4);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#9a3412';
    ctx.fillText(vehicle.notice.text, vehicle.x, vehicle.y - 16);
  }
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
    ctx.fillStyle = '#0f172a';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(zone.name, zone.x, zone.y + 38);
  }

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
    <canvas
      ref={canvasRef}
      className="net-canvas"
      aria-label="Road network with a signal on every incoming road"
    />
  );
}
