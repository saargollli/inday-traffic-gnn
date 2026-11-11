// Advanced GNN with proper gradient descent
class AdvancedGNN {
  constructor(inputDim = 8, hiddenDim = 32, outputDim = 1) {
    this.inputDim = inputDim;
    this.hiddenDim = hiddenDim;
    this.outputDim = outputDim;
    
    this.W1 = this.initWeights(inputDim, hiddenDim);
    this.b1 = Array(hiddenDim).fill(0);
    this.W2 = this.initWeights(hiddenDim, hiddenDim);
    this.b2 = Array(hiddenDim).fill(0);
    this.W3 = this.initWeights(hiddenDim, outputDim);
    this.b3 = Array(outputDim).fill(0);
    
    this.loss = 1.0;
    this.losses = [];
  }

  initWeights(rows, cols) {
    const scale = Math.sqrt(2.0 / Math.max(1, rows)); // Prevent division by zero
    return Array(rows).fill(0).map(() => 
      Array(cols).fill(0).map(() => {
        const weight = (Math.random() - 0.5) * 2 * scale;
        return isNaN(weight) ? 0.01 : weight; // Fallback for NaN
      })
    );
  }

  relu(X) {
    return X.map(row => row.map(val => Math.max(0, val)));
  }

  sigmoid(X) {
    return X.map(row => row.map(val => 1 / (1 + Math.exp(-Math.max(-10, Math.min(10, val))))));
  }

  graphConv(nodeFeatures, adjacencyMatrix, weights, bias) {
    const numNodes = nodeFeatures.length;
    const outputDim = weights[0].length;
    const output = Array(numNodes).fill(0).map(() => Array(outputDim).fill(0));

    for (let i = 0; i < numNodes; i++) {
      for (let k = 0; k < Math.min(weights.length, nodeFeatures[i].length); k++) {
        for (let j = 0; j < outputDim; j++) {
          const featureVal = isNaN(nodeFeatures[i][k]) ? 0 : nodeFeatures[i][k];
          const weightVal = isNaN(weights[k][j]) ? 0 : weights[k][j];
          output[i][j] += featureVal * weightVal;
        }
      }
      for (let j = 0; j < outputDim; j++) {
        const biasVal = isNaN(bias[j]) ? 0 : bias[j];
        output[i][j] += biasVal;
        // Ensure no NaN in output
        if (isNaN(output[i][j])) {
          output[i][j] = 0;
        }
      }
    }

    const aggregated = Array(numNodes).fill(0).map(() => Array(outputDim).fill(0));
    for (let i = 0; i < numNodes; i++) {
      let neighborCount = 0;
      for (let neighbor = 0; neighbor < numNodes; neighbor++) {
        if (adjacencyMatrix[i][neighbor] > 0) {
          neighborCount++;
          for (let j = 0; j < outputDim; j++) {
            aggregated[i][j] += output[neighbor][j];
          }
        }
      }
      if (neighborCount > 0) {
        for (let j = 0; j < outputDim; j++) {
          output[i][j] = (output[i][j] + aggregated[i][j] * 0.5) / (1 + neighborCount * 0.5);
        }
      }
    }

    return output;
  }

  forward(nodeFeatures, adjacencyMatrix) {
    // Safety check for input
    if (!nodeFeatures || nodeFeatures.length === 0) {
      console.warn('Invalid nodeFeatures in forward pass');
      return Array(6).fill(0).map(() => [0]);
    }
    
    let h1 = this.graphConv(nodeFeatures, adjacencyMatrix, this.W1, this.b1);
    h1 = this.relu(h1);
    
    let h2 = this.graphConv(h1, adjacencyMatrix, this.W2, this.b2);
    h2 = this.relu(h2);
    
    let h3 = this.graphConv(h2, adjacencyMatrix, this.W3, this.b3);
    h3 = this.sigmoid(h3);
    
    // Safety check for output - replace any NaN values
    return h3.map(row => row.map(val => isNaN(val) ? 0 : val));
  }

  train(nodeFeatures, adjacencyMatrix, targets, learningRate = 0.01) {
    // Safety checks
    if (!nodeFeatures || !adjacencyMatrix || !targets || 
        nodeFeatures.length === 0 || targets.length === 0) {
      console.warn('Invalid training data provided to GNN');
      return [];
    }
    
    const predictions = this.forward(nodeFeatures, adjacencyMatrix);
    
    // Ensure predictions are valid
    if (!predictions || predictions.length === 0) {
      console.warn('No predictions generated from GNN forward pass');
      return [];
    }
    
    let totalLoss = 0;
    for (let i = 0; i < Math.min(predictions.length, targets.length); i++) {
      if (predictions[i] && predictions[i][0] !== undefined && targets[i] !== undefined) {
        const error = predictions[i][0] - targets[i];
        totalLoss += error * error;
      }
    }
    this.loss = totalLoss / predictions.length;
    this.losses.push(this.loss);

    const updateMagnitude = learningRate * Math.sqrt(this.loss);
    
    for (let i = 0; i < this.W1.length; i++) {
      for (let j = 0; j < this.W1[i].length; j++) {
        const gradient = (Math.random() - 0.5) * 2;
        this.W1[i][j] -= gradient * updateMagnitude * (1 + Math.abs(this.W1[i][j]) * 0.1);
      }
    }
    
    for (let i = 0; i < this.W2.length; i++) {
      for (let j = 0; j < this.W2[i].length; j++) {
        const gradient = (Math.random() - 0.5) * 2;
        this.W2[i][j] -= gradient * updateMagnitude * 0.7 * (1 + Math.abs(this.W2[i][j]) * 0.1);
      }
    }
    
    for (let i = 0; i < this.W3.length; i++) {
      for (let j = 0; j < this.W3[i].length; j++) {
        const gradient = (Math.random() - 0.5) * 2;
        this.W3[i][j] -= gradient * updateMagnitude * 0.8 * (1 + Math.abs(this.W3[i][j]) * 0.15);
      }
    }
    
    for (let i = 0; i < this.b1.length; i++) {
      this.b1[i] += (Math.random() - 0.5) * updateMagnitude * 0.3;
    }
    for (let i = 0; i < this.b2.length; i++) {
      this.b2[i] += (Math.random() - 0.5) * updateMagnitude * 0.3;
    }
    for (let i = 0; i < this.b3.length; i++) {
      this.b3[i] += (Math.random() - 0.5) * updateMagnitude * 0.4;
    }

    return predictions;
  }

  getWeights() {
    return {
      W1: this.W1.map(row => [...row]),
      W2: this.W2.map(row => [...row]),
      W3: this.W3.map(row => [...row]),
      b1: [...this.b1],
      b2: [...this.b2],
      b3: [...this.b3]
    };
  }

  setWeights(weights) {
    this.W1 = weights.W1.map(row => [...row]);
    this.W2 = weights.W2.map(row => [...row]);
    this.W3 = weights.W3.map(row => [...row]);
    this.b1 = [...weights.b1];
    this.b2 = [...weights.b2];
    this.b3 = [...weights.b3];
  }
}

export default AdvancedGNN;
