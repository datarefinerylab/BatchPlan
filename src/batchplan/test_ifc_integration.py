# test_ifc_integration.py
"""
Test script for IFC integration with the new geometry engine
"""

import numpy as np
import trimesh
from geometry_engine import ShapelyTrimeshEngine, IFCGeometryProcessor
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from typing import List

def create_mock_ifc_elements():
    """Create mock IFC-like elements for testing"""
    
    # Create some test meshes that simulate building elements
    elements = []
    
    # Wall 1 - vertical wall from (0,0,0) to (5,0,3)
    wall1 = trimesh.creation.box(extents=[5, 0.2, 3])
    wall1.apply_translation([2.5, 0.1, 1.5])
    # Ensure it's a valid 3D mesh
    assert wall1.is_volume, "Wall1 should be a 3D volume"
    elements.append(("IfcWall", "Wall1", wall1))
    
    # Wall 2 - perpendicular wall
    wall2 = trimesh.creation.box(extents=[0.2, 3, 3])
    wall2.apply_translation([0.1, 1.5, 1.5])
    assert wall2.is_volume, "Wall2 should be a 3D volume"
    elements.append(("IfcWall", "Wall2", wall2))
    
    # Floor slab
    floor = trimesh.creation.box(extents=[6, 4, 0.2])
    floor.apply_translation([3, 2, 0.1])
    assert floor.is_volume, "Floor should be a 3D volume"
    elements.append(("IfcSlab", "Floor1", floor))
    
    # Door opening in wall1
    door = trimesh.creation.box(extents=[1, 0.3, 2.1])
    door.apply_translation([2.5, 0.15, 1.05])
    assert door.is_volume, "Door should be a 3D volume"
    elements.append(("IfcDoor", "Door1", door))
    
    # Window in wall1
    window = trimesh.creation.box(extents=[1.5, 0.25, 1])
    window.apply_translation([1, 0.125, 2])
    assert window.is_volume, "Window should be a 3D volume"
    elements.append(("IfcWindow", "Window1", window))
    
    # Debug: Print bounds for each element
    for elem_type, elem_name, mesh in elements:
        print(f"{elem_name} bounds: {mesh.bounds} (shape: {mesh.bounds.shape})")
        print(f"{elem_name} is_volume: {mesh.is_volume}")
    
    return elements

def debug_mesh_properties(mesh, name="mesh"):
    """Debug function to inspect mesh properties"""
    print(f"\n=== Debug info for {name} ===")
    print(f"Type: {type(mesh)}")
    print(f"Bounds: {mesh.bounds}")
    print(f"Bounds shape: {mesh.bounds.shape}")
    print(f"Vertices shape: {mesh.vertices.shape}")
    print(f"Faces shape: {mesh.faces.shape}")
    print(f"Is volume: {mesh.is_volume}")
    print(f"Is watertight: {mesh.is_watertight}")
    print(f"Volume: {mesh.volume}")
    print(f"Area: {mesh.area}")
    print(f"Extents: {mesh.extents}")
    print("=" * (15 + len(name)))

def test_floor_plan_extraction():
    """Test extracting floor plans at different heights"""
    
    print("Testing floor plan extraction...")
    
    # Create mock elements
    elements = create_mock_ifc_elements()
    
    # Debug the first element
    if elements:
        debug_mesh_properties(elements[0][2], elements[0][1])
    
    # Initialize processor
    processor = IFCGeometryProcessor()
    
    # Test different heights
    test_heights = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    results = {}
    
    for height in test_heights:
        print(f"Testing height: {height}m")
        
        # Extract polygons at this height
        polygons = []
        for elem_type, elem_name, mesh in elements:
            plane_origin = (0, 0, height)
            plane_normal = (0, 0, 1)
            
            # Check if element intersects with the plane
            bounds = mesh.bounds
            tolerance = 0.1
            
            # Handle both 2D and 3D bounds
            if len(bounds) == 6:  # 3D bounds: [xmin, ymin, zmin, xmax, ymax, zmax]
                zmin, zmax = bounds[2], bounds[5]
            elif len(bounds) == 4:  # 2D bounds: [xmin, ymin, xmax, ymax] - assume z=0
                zmin, zmax = 0, 0
            else:
                print(f"Unexpected bounds format: {bounds}")
                continue
                
            if zmin <= height + tolerance and zmax >= height - tolerance:
                element_polygons = processor.engine.intersect_with_plane(
                    mesh, plane_origin, plane_normal
                )
                
                if element_polygons:
                    for poly in element_polygons:
                        polygons.append((elem_type, elem_name, poly))
        
        results[height] = polygons
        print(f"  Found {len(polygons)} intersections")
        
        # Calculate total area
        total_area = sum(poly.area for _, _, poly in polygons)
        print(f"  Total area: {total_area:.2f} m²")
    
    return results

def visualize_floor_plan(polygons: List[tuple], height: float, save_path: str = None):
    """Visualize a floor plan"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Color mapping for different element types
    colors = {
        'IfcWall': 'black',
        'IfcSlab': 'lightgray',
        'IfcDoor': 'brown',
        'IfcWindow': 'lightblue',
        'IfcColumn': 'darkgray'
    }
    
    # Plot each polygon
    for elem_type, elem_name, poly in polygons:
        if poly is None or poly.is_empty:
            continue
            
        color = colors.get(elem_type, 'blue')
        
        # Get coordinates
        x, y = poly.exterior.xy
        ax.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add label
        centroid = poly.centroid
        ax.text(centroid.x, centroid.y, elem_name, ha='center', va='center', 
                fontsize=8, weight='bold')
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Floor Plan at Height {height}m')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, edgecolor='black') 
                      for elem_type, color in colors.items()]
    ax.legend(legend_elements, colors.keys(), loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved floor plan to {save_path}")
    
    plt.show()

def test_vs_original_output():
    """Test that mimics the original BatchPlan workflow"""
    
    print("Testing against original BatchPlan workflow...")
    
    # Create test scenario
    elements = create_mock_ifc_elements()
    
    # Initialize processor
    processor = IFCGeometryProcessor()
    
    # Simulate the original get_section_faces workflow
    section_height = 1.5  # Middle height
    
    # This simulates what the original extract_floor_plans.py does:
    # 1. Create section surface at height
    # 2. Intersect each shape with the surface  
    # 3. Extract faces/polygons
    
    print(f"Extracting section at height {section_height}m")
    
    section_polygons = []
    
    for elem_type, elem_name, mesh in elements:
        # This replaces the original get_section_faces function
        polygons = processor.engine.intersect_with_plane(
            mesh, 
            plane_origin=(0, 0, section_height),
            plane_normal=(0, 0, 1)
        )
        
        for poly in polygons:
            section_polygons.append((elem_type, elem_name, poly))
            print(f"  {elem_type} {elem_name}: {poly.area:.3f} m²")
    
    print(f"Total elements in section: {len(section_polygons)}")
    
    # Visualize the result
    visualize_floor_plan(section_polygons, section_height, 
                        f"test_floor_plan_{section_height}m.png")
    
    return section_polygons

def benchmark_performance():
    """Benchmark the performance of the new engine"""
    
    import time
    
    print("Benchmarking performance...")
    
    # Create larger test scenario
    elements = []
    
    # Create grid of walls
    for i in range(10):
        for j in range(10):
            wall = trimesh.creation.box(extents=[0.2, 1, 3])
            wall.apply_translation([i * 2, j * 2, 1.5])
            elements.append(("IfcWall", f"Wall_{i}_{j}", wall))
    
    print(f"Created {len(elements)} test elements")
    
    # Time the intersection process
    processor = IFCGeometryProcessor()
    
    start_time = time.time()
    
    polygons = []
    for elem_type, elem_name, mesh in elements:
        poly_list = processor.engine.intersect_with_plane(
            mesh, 
            plane_origin=(0, 0, 1.5),
            plane_normal=(0, 0, 1)
        )
        polygons.extend(poly_list)
    
    end_time = time.time()
    
    print(f"Processed {len(elements)} elements in {end_time - start_time:.3f} seconds")
    print(f"Found {len(polygons)} intersection polygons")
    print(f"Average time per element: {(end_time - start_time) / len(elements) * 1000:.2f} ms")

def main():
    """Run all tests"""
    
    print("=== Geometry Engine Test Suite ===\n")
    
    # Test 1: Basic floor plan extraction
    results = test_floor_plan_extraction()
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Visualize a specific floor plan
    test_vs_original_output()
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: Performance benchmark
    benchmark_performance()
    
    print("\n=== All tests completed ===")

if __name__ == "__main__":
    main()
