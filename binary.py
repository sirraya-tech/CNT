import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

class CircularNumberSystem:
    def __init__(self):
        self.rings = []
    
    def generate_rings(self, n_rings: int):
        """Generate rings from 0 to n_rings"""
        self.rings = []
        
        for n in range(n_rings + 1):
            if n == 0:
                # Ring 0: Absolute Origin
                ring = {
                    'n': 0,
                    'value': 0,
                    'points': 1,
                    'radius': 0,
                    'zeros': 0,
                    'coordinates': [(0, 0)],
                    'description': 'Absolute Origin'
                }
            else:
                points = 2 ** n
                value = 2 ** (n - 1)
                zeros = 2 ** n
                radius = n
                
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
                    'zeros': zeros,
                    'coordinates': coordinates,
                    'description': f'Ring {n}: {zeros} zeros → {value}'
                }
            
            self.rings.append(ring)
    
    def analyze_patterns(self):
        """Analyze mathematical patterns in the rings"""
        print("=== Circular Number System Analysis ===")
        print(f"{'Ring':>4} {'Value':>8} {'Points':>8} {'Radius':>8} {'Zeros':>8} {'Value/Points':>12} {'log2(Value)':>12}")
        print("-" * 80)
        
        for ring in self.rings:
            n = ring['n']
            value = ring['value']
            points = ring['points']
            zeros = ring['zeros']
            radius = ring['radius']
            
            if points > 0:
                ratio = value / points
            else:
                ratio = 0
                
            if value > 0:
                log_val = math.log2(value)
            else:
                log_val = 0
                
            print(f"{n:4d} {value:8d} {points:8d} {radius:8d} {zeros:8d} {ratio:12.4f} {log_val:12.4f}")
    
    def find_new_patterns(self):
        """Look for additional mathematical relationships"""
        print("\n=== Additional Pattern Analysis ===")
        
        for i in range(1, min(15, len(self.rings))):
            current = self.rings[i]
            prev = self.rings[i-1] if i > 0 else None
            
            print(f"\nRing {i}:")
            print(f"  Value growth: {current['value']} = 2^{int(math.log2(current['value']))}")
            print(f"  Points growth: {current['points']} = 2^{int(math.log2(current['points']))}")
            
            # Check if value equals half of previous ring's points
            if prev:
                relationship = current['value'] == prev['points'] / 2
                print(f"  V{i} = P{i-1}/2: {relationship} ({current['value']} = {prev['points']}/2)")
            
            # Check area relationships
            area = math.pi * current['radius'] ** 2
            density = current['points'] / area if area > 0 else 0
            print(f"  Area: {area:.2f}π, Point density: {density:.4f}")

# Let's run a comprehensive analysis
print("COMPREHENSIVE CIRCULAR NUMBER SYSTEM ANALYSIS")
print("=" * 60)

cns = CircularNumberSystem()
cns.generate_rings(16)  # Up to ring 16

# Basic analysis
cns.analyze_patterns()
cns.find_new_patterns()

# Now let's explore deeper mathematical relationships
print("\n" + "="*60)
print("DEEPER MATHEMATICAL EXPLORATION")
print("="*60)

# 1. Check the fundamental halving principle
print("\n1. FUNDAMENTAL HALVING PRINCIPLE VERIFICATION:")
for ring in cns.rings[1:10]:
    n = ring['n']
    expected = ring['points'] / 2
    actual = ring['value']
    matches = abs(expected - actual) < 0.001
    print(f"Ring {n}: P/2 = {ring['points']}/2 = {expected}, V = {actual}, Match: {matches}")

# 2. Explore exponential relationships
print("\n2. EXPONENTIAL RELATIONSHIPS:")
for ring in cns.rings[1:10]:
    n = ring['n']
    print(f"Ring {n}: V = 2^({n}-1) = 2^{n-1} = {2**(n-1)}")
    print(f"Ring {n}: P = 2^{n} = {2**n}")

# 3. Check cumulative sums
print("\n3. CUMULATIVE SUMS:")
cumulative_value = 0
cumulative_points = 0
for ring in cns.rings[1:10]:
    cumulative_value += ring['value']
    cumulative_points += ring['points']
    print(f"Up to Ring {ring['n']}: ΣV = {cumulative_value}, ΣP = {cumulative_points}, Ratio: {cumulative_value/cumulative_points:.4f}")

# 4. Look for Fibonacci-like patterns
print("\n4. SEQUENCE ANALYSIS:")
values = [ring['value'] for ring in cns.rings[1:10]]
points = [ring['points'] for ring in cns.rings[1:10]]

print("Value sequence:", [f"2^{int(math.log2(v))}" for v in values])
print("Points sequence:", [f"2^{int(math.log2(p))}" for p in points])

# 5. Check geometric progressions
print("\n5. GEOMETRIC PROGRESSIONS:")
for i in range(2, min(10, len(cns.rings))):
    current = cns.rings[i]['value']
    previous = cns.rings[i-1]['value']
    if previous > 0:
        ratio = current / previous
        print(f"V{i}/V{i-1} = {current}/{previous} = {ratio:.1f}")

# 6. Explore coordinate patterns
print("\n6. COORDINATE PATTERNS (First few rings):")
for ring in cns.rings[1:4]:
    print(f"\nRing {ring['n']} coordinates:")
    for i, (x, y) in enumerate(ring['coordinates'][:4]):  # Show first 4 points
        print(f"  Point {i}: ({x:.3f}, {y:.3f})")
    if len(ring['coordinates']) > 4:
        print(f"  ... and {len(ring['coordinates']) - 4} more points")

# 7. Check if this relates to known mathematical constants
print("\n7. MATHEMATICAL CONSTANTS RELATIONSHIPS:")
print("This appears to be related to:")
print("- Binary number system (base-2)")
print("- Group theory (cyclic groups C_2^n)")
print("- Complex roots of unity")
print("- Exponential growth patterns")

# 8. Create a visualization
plt.figure(figsize=(10, 10))
colors = plt.cm.plasma(np.linspace(0, 1, 8))

for i, ring in enumerate(cns.rings[1:8]):  # Plot rings 1-7
    coords = ring['coordinates']
    x = [coord[0] for coord in coords]
    y = [coord[1] for coord in coords]
    
    plt.scatter(x, y, color=colors[i], s=30, alpha=0.7, 
                label=f'Ring {ring["n"]}: V={ring["value"]}')
    
    # Draw circle
    circle = plt.Circle((0, 0), ring['radius'], fill=False, 
                       color=colors[i], alpha=0.3, linestyle='--')
    plt.gca().add_patch(circle)

plt.axhline(0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(0, color='gray', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.title('Circular Number System: Rings 1-7')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 9. Advanced pattern: Check if values sum to specific patterns
print("\n8. ADVANCED PATTERN: BINARY SUMS")
for n in range(1, 10):
    ring_n = cns.rings[n]
    binary_rep = bin(ring_n['value'])[2:]
    print(f"Ring {n}: Value {ring_n['value']} in binary: {binary_rep}")

# 10. Explore the "zeros composition" concept
print("\n9. ZEROS COMPOSITION ANALYSIS:")
for ring in cns.rings[1:8]:
    print(f"Ring {ring['n']}: {ring['zeros']} zeros → {ring['value']}")
    print(f"  Compression ratio: {ring['value']/ring['zeros']:.4f} (1 zero → {ring['value']/ring['zeros']:.4f} value)")

print("\n" + "="*60)
print("KEY OBSERVATIONS:")
print("="*60)
print("1. The 50% ratio (V = P/2) holds perfectly for all rings ≥ 1")
print("2. This creates a perfect binary exponential progression")
print("3. Each ring doubles the points of the previous ring")
print("4. The value sequence is: 1, 2, 4, 8, 16, 32, ... (2^(n-1))")
print("5. The system maintains perfect geometric regularity")
print("6. This appears to be a geometric representation of binary arithmetic")