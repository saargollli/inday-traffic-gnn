export const federatedAggregation = (zones, adjacency) => {
  console.log('Starting federated aggregation...');
  
  zones.forEach(zone => {
    // Find connected zones (neighbors in the graph)
    const connectedIds = [];
    for (let i = 0; i < adjacency.length; i++) {
      if (adjacency[zone.id][i] > 0 || adjacency[i][zone.id] > 0) {
        connectedIds.push(i);
      }
    }
    
    if (connectedIds.length === 0) {
      console.log(`Zone ${zone.id}: No connected zones, keeping local model`);
      return;
    }

    // Only include this zone and its connected neighbors (federated learning principle)
    const participants = [zone, ...zones.filter(z => connectedIds.includes(z.id))];
    
    console.log(`Zone ${zone.id}: Aggregating with ${connectedIds.length} connected zones: [${connectedIds.join(', ')}]`);
    
    // Initialize averaged weights structure
    const avgWeights = {
      W1: zone.gnn.W1.map(row => row.map(() => 0)),
      W2: zone.gnn.W2.map(row => row.map(() => 0)),
      W3: zone.gnn.W3.map(row => row.map(() => 0)),
      b1: zone.gnn.b1.map(() => 0),
      b2: zone.gnn.b2.map(() => 0),
      b3: zone.gnn.b3.map(() => 0)
    };

    // Aggregate weights from all participating zones (federated averaging)
    participants.forEach(z => {
      const w = z.gnn.getWeights();
      ['W1', 'W2', 'W3'].forEach(layer => {
        for (let i = 0; i < w[layer].length; i++) {
          for (let j = 0; j < w[layer][i].length; j++) {
            avgWeights[layer][i][j] += w[layer][i][j];
          }
        }
      });
      ['b1', 'b2', 'b3'].forEach(layer => {
        for (let i = 0; i < w[layer].length; i++) {
          avgWeights[layer][i] += w[layer][i];
        }
      });
    });

    // Average the weights (FedAvg algorithm)
    const n = participants.length;
    ['W1', 'W2', 'W3'].forEach(layer => {
      for (let i = 0; i < avgWeights[layer].length; i++) {
        for (let j = 0; j < avgWeights[layer][i].length; j++) {
          avgWeights[layer][i][j] /= n;
        }
      }
    });
    ['b1', 'b2', 'b3'].forEach(layer => {
      for (let i = 0; i < avgWeights[layer].length; i++) {
        avgWeights[layer][i] /= n;
      }
    });

    // Update this zone's model with the averaged weights
    zone.gnn.setWeights(avgWeights);
  });
  
  console.log('Federated aggregation completed');
};
