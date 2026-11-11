// One scale for the lanes and the busiest-signal list.
// An empty road stays gray. Cars on the road turn it green, then amber, then red
// as the road fills toward its capacity.

export const TRAFFIC_LEVELS = [
  { id: 'empty', label: 'Empty', color: '#b7c3d0', tint: '#f3f6f9', text: '#475569' },
  { id: 'flowing', label: 'Cars moving', color: '#1f9d62', tint: '#e7f6ee', text: '#14532d' },
  { id: 'building', label: 'Filling up', color: '#e09a00', tint: '#fff6df', text: '#92400e' },
  { id: 'jammed', label: 'Jammed', color: '#dc4c4c', tint: '#fdecec', text: '#991b1b' }
];

const RANK = { empty: 0, flowing: 1, building: 2, jammed: 3 };

export function trafficLevel(vehicleCount, load) {
  if (!vehicleCount) return TRAFFIC_LEVELS[0];
  if (load >= 0.66) return TRAFFIC_LEVELS[3];
  if (load >= 0.35) return TRAFFIC_LEVELS[2];
  return TRAFFIC_LEVELS[1];
}

export function levelRank(id) {
  return RANK[id] || 0;
}
