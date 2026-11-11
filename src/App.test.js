import { render, screen } from '@testing-library/react';
import App from './App';

test('shows the study title', () => {
  render(<App />);
  expect(screen.getByRole('heading', { name: /federated learning/i })).toBeInTheDocument();
});
