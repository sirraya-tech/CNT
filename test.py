import math
import matplotlib.pyplot as plt
import numpy as np
import sympy
from typing import List, Tuple

class PrimeCircularSystem:
    def __init__(self):
        self.rings = []
        self.primes = []
    
    def generate_primes_up_to(self, n: int):
        """Generate list of primes up to n"""
        self.primes = list(sympy.primerange(0, n+1))
        return self.primes
    
    def generate_prime_rings(self, n_rings: int, mode='prime_values'):
        """
        Generate rings using prime numbers in different ways
        """
        self.rings = []
        # Generate enough primes
        primes = self.generate_primes_up_to(1000)
        
        for n in range(n_rings + 1):
            if n == 0:
                # Ring 0: Absolute Origin
                ring = {
                    'n': 0,
                    'value': 0,
                    'points': 1,
                    'radius': 0,
                    'prime_value': None,
                    'coordinates': [(0, 0)],
                    'description': 'Absolute Origin'
                }
                self.rings.append(ring)
                continue
            
            # Initialize variables with defaults
            value = 0
            points = 0
            radius = n
            prime_val = None
            
            try:
                if mode == 'prime_values':
                    # Use nth prime as value
                    if n-1 < len(primes):
                        prime_val = primes[n-1]
                        value = prime_val
                        points = 2 * n  # Simple scaling
                    else:
                        continue
                
                elif mode == 'prime_points':
                    # Use nth prime to determine points
                    if n-1 < len(primes):
                        prime_val = primes[n-1]
                        points = prime_val
                        value = points // 2 if points >= 2 else 1
                    else:
                        continue
                
                elif mode == 'prime_radius':
                    # Use nth prime as radius
                    if n-1 < len(primes):
                        prime_val = primes[n-1]
                        radius = prime_val
                        points = 2 * n
                        value = n
                    else:
                        continue
                
                elif mode == 'prime_index':
                    # Only create rings at prime indices
                    if sympy.isprime(n):
                        prime_val = n
                        points = 2 ** n
                        value = 2 ** (n - 1)
                    else:
                        continue
                
                elif mode == 'hybrid':
                    # Mix of binary and prime patterns
                    points = 2 ** n
                    if n-1 < len(primes):
                        prime_val = primes[n-1]
                        value = prime_val
                    else:
                        value = 2 ** (n - 1)
                
                # Generate coordinates
                coordinates = []
                for k in range(points):
                    angle = 2 * math.pi * k / points
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    coordinates.append((x, y))
                
                ring = {
                    'n': n,
                    'value': value,
                    'points': points,
                    'radius': radius,
                    'prime_value': prime_val,
                    'coordinates': coordinates,
                    'description': f'Ring {n}: Prime {prime_val if prime_val else "N/A"}'
                }
                
                self.rings.append(ring)
                
            except Exception as e:
                print(f"Error generating ring {n}: {e}")
                continue
    
    def analyze_prime_patterns(self):
        """Analyze mathematical patterns in prime rings"""
        if not self.rings:
            print("No rings generated!")
            return
            
        print("=== PRIME CIRCULAR SYSTEM ANALYSIS ===")
        print(f"{'Ring':>4} {'Value':>8} {'Points':>8} {'Radius':>8} {'Prime':>8} {'Value/Points':>12} {'IsPrimeVal':>12}")
        print("-" * 90)
        
        for ring in self.rings:
            n = ring['n']
            value = ring['value']
            points = ring['points']
            radius = ring['radius']
            prime_val = ring['prime_value']
            
            if points > 0:
                ratio = value / points
            else:
                ratio = 0
                
            is_prime_val = sympy.isprime(value) if value > 1 else False
            
            prime_str = str(prime_val) if prime_val is not None else "N/A"
            
            print(f"{n:4d} {value:8d} {points:8d} {radius:8d} {prime_str:>8} {ratio:12.4f} {str(is_prime_val):>12}")

def create_prime_visualization(prime_system, title):
    """Create visualization for prime-based system"""
    if len(prime_system.rings) <= 1:
        print(f"Not enough rings to visualize for {title}")
        return
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(prime_system.rings)))
    
    # Plot 1: Circular rings
    for i, ring in enumerate(prime_system.rings[1:], 1):  # Skip ring 0
        coords = ring['coordinates']
        if not coords:
            continue
        x = [coord[0] for coord in coords]
        y = [coord[1] for coord in coords]
        
        # Color based on whether value is prime
        color = 'red' if sympy.isprime(ring['value']) else colors[i-1]
        
        ax1.scatter(x, y, color=color, s=30, alpha=0.7, 
                   label=f'Ring {ring["n"]}')
        
        circle = plt.Circle((0, 0), ring['radius'], fill=False, 
                           color=color, alpha=0.3, linestyle='--')
        ax1.add_patch(circle)
    
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{title}\n(Red = Prime Value Rings)', fontweight='bold')
    ax1.legend()
    
    # Plot 2: Value progression
    rings = [ring['n'] for ring in prime_system.rings[1:]]
    values = [ring['value'] for ring in prime_system.rings[1:]]
    
    # Color points based on primality
    for i, (r, v) in enumerate(zip(rings, values)):
        color = 'red' if sympy.isprime(v) else 'blue'
        ax2.scatter(r, v, color=color, s=50, zorder=3)
    
    ax2.plot(rings, values, 'k-', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Ring Number')
    ax2.set_ylabel('Value')
    ax2.set_title('Value Progression\n(Red = Prime Values)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Ratio analysis
    ratios = []
    valid_rings = []
    for ring in prime_system.rings[1:]:
        if ring['points'] > 0:
            ratios.append(ring['value'] / ring['points'])
            valid_rings.append(ring['n'])
    
    if ratios:
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Reference')
        ax3.plot(valid_rings, ratios, 'go-', linewidth=2, markersize=6)
        ax3.set_xlabel('Ring Number')
        ax3.set_ylabel('Value/Points Ratio')
        ax3.set_title('Ratio Analysis', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Prime distribution
    prime_rings = [ring['n'] for ring in prime_system.rings[1:] if sympy.isprime(ring['value'])]
    if prime_rings:
        prime_density = [1] * len(prime_rings)  # Mark prime rings
        ax4.scatter(prime_rings, prime_density, color='red', s=100, alpha=0.7)
        ax4.set_xlabel('Ring Number')
        ax4.set_yticks([])
        ax4.set_title(f'Prime Distribution\n({len(prime_rings)} prime-value rings)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_prime_patterns_comparison():
    """Compare different prime integration methods"""
    print("COMPREHENSIVE PRIME PATTERN COMPARISON")
    print("=" * 60)
    
    modes = [
        ('prime_values', 'Prime Numbers as Ring Values'),
        ('prime_points', 'Primes Determine Points'), 
        ('prime_radius', 'Primes as Radii'),
        ('prime_index', 'Prime-Numbered Rings Only'),
        ('hybrid', 'Hybrid Binary-Prime System')
    ]
    
    results = {}
    
    for mode, description in modes:
        print(f"\n{'='*50}")
        print(f"ANALYZING: {description}")
        print(f"{'='*50}")
        
        system = PrimeCircularSystem()
        system.generate_prime_rings(12, mode=mode)
        
        if system.rings:
            system.analyze_prime_patterns()
            results[mode] = system
            
            # Calculate statistics
            if len(system.rings) > 1:
                values = [ring['value'] for ring in system.rings[1:]]
                points = [ring['points'] for ring in system.rings[1:]]
                ratios = [v/p for v, p in zip(values, points) if p > 0]
                
                if ratios:
                    prime_count = sum(1 for v in values if sympy.isprime(v) and v > 1)
                    print(f"📊 Statistics: {prime_count} prime values, Avg ratio: {np.mean(ratios):.4f}")
        
            # Create visualization for key modes
            if mode in ['prime_values', 'hybrid']:
                create_prime_visualization(system, description)
        else:
            print("No rings generated for this mode!")
    
    return results

def investigate_prime_properties():
    """Deep investigation of prime properties in circular systems"""
    print("\n" + "="*70)
    print("DEEP PRIME PROPERTIES INVESTIGATION")
    print("="*70)
    
    # Focus on prime values mode
    system = PrimeCircularSystem()
    system.generate_prime_rings(20, mode='prime_values')
    
    if len(system.rings) > 1:
        print("\nPRIME GAP ANALYSIS:")
        prime_values = [ring['prime_value'] for ring in system.rings[1:] if ring['prime_value'] is not None]
        prime_gaps = [prime_values[i+1] - prime_values[i] for i in range(len(prime_values)-1)]
        
        print(f"Primes used: {prime_values[:10]}...")  # Show first 10
        print(f"Prime gaps: {prime_gaps[:10]}...")
        print(f"Average gap: {np.mean(prime_gaps):.2f}")
        
        # Twin primes analysis
        twin_primes = []
        for i in range(len(prime_gaps)):
            if prime_gaps[i] == 2:
                twin_primes.append((prime_values[i], prime_values[i+1]))
        
        print(f"Twin primes found: {twin_primes}")
        
        # Prime density by ring
        print("\nPRIME DENSITY BY RING:")
        for ring in system.rings[1:11]:  # First 10 rings
            if ring['prime_value']:
                density = ring['prime_value'] / ring['points'] if ring['points'] > 0 else 0
                is_value_prime = sympy.isprime(ring['value'])
                prime_mark = "✓" if is_value_prime else "✗"
                print(f"Ring {ring['n']:2d}: Prime {ring['prime_value']:2d}, Points {ring['points']:3d}, Density {density:.3f} {prime_mark}")

def create_comparison_visualization():
    """Compare original binary system with prime systems"""
    print("\n" + "="*70)
    print("BINARY vs PRIME SYSTEM COMPARISON")
    print("="*70)
    
    # Original binary system
    binary_system = PrimeCircularSystem()
    binary_system.generate_prime_rings(8, mode='hybrid')  # Use hybrid as proxy for binary
    
    # Prime value system
    prime_system = PrimeCircularSystem()
    prime_system.generate_prime_rings(8, mode='prime_values')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot binary system
    colors_binary = plt.cm.Blues(np.linspace(0.3, 1, len(binary_system.rings)))
    for i, ring in enumerate(binary_system.rings[1:], 1):
        coords = ring['coordinates']
        if coords:
            x = [coord[0] for coord in coords]
            y = [coord[1] for coord in coords]
            ax1.scatter(x, y, color=colors_binary[i-1], s=30, alpha=0.7)
            circle = plt.Circle((0, 0), ring['radius'], fill=False, 
                               color=colors_binary[i-1], alpha=0.3, linestyle='--')
            ax1.add_patch(circle)
    
    ax1.set_aspect('equal')
    ax1.set_title('Binary System\n(Perfect Symmetry)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot prime system
    for i, ring in enumerate(prime_system.rings[1:], 1):
        coords = ring['coordinates']
        if coords:
            x = [coord[0] for coord in coords]
            y = [coord[1] for coord in coords]
            color = 'red' if sympy.isprime(ring['value']) else 'green'
            ax2.scatter(x, y, color=color, s=30, alpha=0.7)
            circle = plt.Circle((0, 0), ring['radius'], fill=False, 
                               color=color, alpha=0.3, linestyle='--')
            ax2.add_patch(circle)
    
    ax2.set_aspect('equal')
    ax2.set_title('Prime Value System\n(Red = Prime Values)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Numerical comparison
    print("\nNUMERICAL COMPARISON:")
    print(f"{'Ring':>4} {'Binary Value':>12} {'Prime Value':>12} {'Binary Ratio':>12} {'Prime Ratio':>12}")
    print("-" * 70)
    
    for i in range(1, min(len(binary_system.rings), len(prime_system.rings))):
        b_ring = binary_system.rings[i]
        p_ring = prime_system.rings[i]
        
        b_ratio = b_ring['value'] / b_ring['points'] if b_ring['points'] > 0 else 0
        p_ratio = p_ring['value'] / p_ring['points'] if p_ring['points'] > 0 else 0
        
        print(f"{i:4d} {b_ring['value']:12d} {p_ring['value']:12d} {b_ratio:12.4f} {p_ratio:12.4f}")

def run_complete_prime_analysis():
    """Main function to run complete prime analysis"""
    print("🔢 PRIME CIRCULAR NUMBER SYSTEM EXPLORER")
    print("=" * 60)
    print("Discovering how primes behave in circular geometry!")
    print("Comparing order (binary) vs complexity (primes)...\n")
    
    # 1. Compare different prime integration methods
    results = analyze_prime_patterns_comparison()
    
    # 2. Deep prime properties investigation
    investigate_prime_properties()
    
    # 3. Binary vs Prime comparison
    create_comparison_visualization()
    
    print("\n" + "="*70)
  

# Run the complete analysis
if __name__ == "__main__":
    run_complete_prime_analysis()