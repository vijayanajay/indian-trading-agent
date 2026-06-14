import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MarketOverview } from './MarketOverview';
import * as api from '@/lib/api';

// Mock the API module
vi.mock('@/lib/api', () => ({
  getMarketStatus: vi.fn(),
}));

describe('MarketOverview Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders loading skeleton initially', () => {
    // Mock the API to never resolve so it stays in loading state
    vi.mocked(api.getMarketStatus).mockReturnValue(new Promise(() => {}));

    const { container } = render(<MarketOverview />);

    // Check if the loading skeletons are rendered
    const skeletons = container.querySelectorAll('.animate-pulse > div');
    expect(skeletons.length).toBe(2);
  });

  it('renders market data correctly when API call is successful', async () => {
    const mockData = {
      session: 'open',
      nifty: { price: 22000.50, change: 150.25, change_percent: 0.68 },
      banknifty: { price: 47000.75, change: -50.10, change_percent: -0.11 }
    };
    vi.mocked(api.getMarketStatus).mockResolvedValue(mockData);

    render(<MarketOverview />);

    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText('OPEN')).toBeInTheDocument();
    });

    // Check NIFTY values
    expect(screen.getByText('NIFTY 50')).toBeInTheDocument();
    expect(screen.getByText('22,000.5')).toBeInTheDocument();
    expect(screen.getByText('+150.25 (+0.68%)')).toBeInTheDocument();
    expect(screen.getByText('+150.25 (+0.68%)')).toHaveClass('text-green-400');

    // Check BANK NIFTY values
    expect(screen.getByText('BANK NIFTY')).toBeInTheDocument();
    expect(screen.getByText('47,000.75')).toBeInTheDocument();
    expect(screen.getByText('-50.10 (-0.11%)')).toBeInTheDocument();
    expect(screen.getByText('-50.10 (-0.11%)')).toHaveClass('text-red-400');
  });

  it('renders correctly during post-market session', async () => {
    const mockData = {
      session: 'post_market',
      nifty: { price: 21000, change: -200, change_percent: -0.95 },
      banknifty: { price: 45000, change: 100, change_percent: 0.22 }
    };
    vi.mocked(api.getMarketStatus).mockResolvedValue(mockData);

    render(<MarketOverview />);

    await waitFor(() => {
      expect(screen.getByText('POST MARKET')).toBeInTheDocument();
    });
  });

  it('renders closed session badge gracefully when invalid session passed', async () => {
    const mockData = {
      session: 'invalid_session_type',
      nifty: { price: 20000, change: 0, change_percent: 0 },
      banknifty: { price: 40000, change: 0, change_percent: 0 }
    };
    vi.mocked(api.getMarketStatus).mockResolvedValue(mockData);

    render(<MarketOverview />);

    await waitFor(() => {
      expect(screen.getByText('INVALID SESSION_TYPE')).toBeInTheDocument();
    });

    // Check if it falls back to closed colors if not found in sessionColors mapping
    // But since the badge class directly does `sessionColors[data.session] || sessionColors.closed`
    const badge = screen.getByText('INVALID SESSION_TYPE');
    expect(badge).toHaveClass('bg-red-500/20 text-red-400');
  });

  it('handles API failure gracefully (stays in loading state or throws handled error)', async () => {
    // If API fails, `setData` is not called and it silently ignores
    // Meaning it should stay in the loading state forever in this implementation
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.mocked(api.getMarketStatus).mockRejectedValue(new Error('Network error'));

    const { container } = render(<MarketOverview />);

    // Wait a little to let the promise rejection process
    await new Promise(r => setTimeout(r, 50));

    // It should still have the loading skeleton
    const skeletons = container.querySelectorAll('.animate-pulse > div');
    expect(skeletons.length).toBe(2);

    consoleErrorSpy.mockRestore();
  });
});
