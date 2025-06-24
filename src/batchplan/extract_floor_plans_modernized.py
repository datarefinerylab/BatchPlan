# extract_floor_plans_modernized.py
"""
Modernized floor plan extraction using shapely+trimesh instead of pythonocc-core
This replaces the original extract_floor_plans.py with headless operation
"""

import argparse
import glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Optional

import ifcopenshell
import ifcopenshell.geom
from ifcopenshell.util.element import get_decomposition

from geometry_engine import ShapelyTrimeshEngine, IFCGeometryProcessor, create_geometry_engine
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ModernFormatter:
    """Base class for modern formatters (headless)"""
    
    def __init__(self, context):
        self.context = context
        self.engine = context.get('engine', create_geometry_engine())
    
    def process(self, name: str, elements: list, polygons: List[Polygon]):
        """Process a floor level - to be implemented by subclasses"""
        raise NotImplementedError()


class FloorPlanImageFormatter(ModernFormatter):
    """Generate floor plan images using matplotlib (headless)"""
    
    def __init__(self, context):
        super().__init__(context)
        style = context.get('style', 'professional')
        
        if style == 'professional':
            self._setup_professional_style()
        elif style == 'minimal':
            self._setup_minimal_style()
        elif style == 'colorful':
            self._setup_colorful_style()
        elif style == 'technical':
            self._setup_technical_style()
    
    def _setup_professional_style(self):
        """Professional architectural color scheme"""
        self.colors = {
            'IfcWall': '#2C3E50',           # Dark blue-gray for walls
            'IfcWallStandardCase': '#2C3E50',
            'IfcSlab': '#ECF0F1',           # Light gray for slabs/floors
            'IfcColumn': '#34495E',         # Darker gray for columns
            'IfcBeam': '#8B4513',           # Brown for beams
            'IfcDoor': '#E67E22',           # Orange for doors
            'IfcWindow': '#3498DB',         # Blue for windows
            'IfcStair': '#9B59B6',          # Purple for stairs
            'IfcStairFlight': '#9B59B6',
            'IfcRailing': '#95A5A6',        # Medium gray for railings
            'IfcRamp': '#F39C12',           # Yellow for ramps
            'IfcFurnishingElement': '#16A085', # Teal for furniture
            'IfcBuildingElementProxy': '#7F8C8D', # Gray for proxies
            'IfcCovering': '#D5DBDB',       # Very light gray for coverings
            'IfcFlowTerminal': '#E74C3C',   # Red for MEP terminals
            'IfcDistributionElement': '#C0392B', # Dark red for MEP
            'IfcSpace': '#F8F9FA',          # Almost white for spaces
            'IfcZone': '#F1C40F',           # Yellow for zones
        }
        self.line_weights = {
            'IfcWall': 2.0, 'IfcWallStandardCase': 2.0, 'IfcSlab': 0.5,
            'IfcColumn': 2.0, 'IfcBeam': 1.5, 'IfcDoor': 1.0, 'IfcWindow': 1.0,
            'IfcStair': 1.5, 'IfcStairFlight': 1.5, 'IfcRailing': 0.8, 'default': 0.5
        }
        self.alphas = {
            'IfcWall': 0.9, 'IfcWallStandardCase': 0.9, 'IfcSlab': 0.3,
            'IfcColumn': 0.9, 'IfcBeam': 0.8, 'IfcDoor': 0.8, 'IfcWindow': 0.7,
            'IfcStair': 0.8, 'IfcSpace': 0.1, 'default': 0.7
        }
    
    def _setup_minimal_style(self):
        """Clean minimal black and white style"""
        base_color = '#2C3E50'
        light_color = '#ECF0F1'
        
        self.colors = {
            'IfcWall': base_color, 'IfcWallStandardCase': base_color,
            'IfcSlab': light_color, 'IfcColumn': base_color, 'IfcBeam': base_color,
            'IfcDoor': '#95A5A6', 'IfcWindow': '#BDC3C7', 'IfcStair': base_color,
            'IfcStairFlight': base_color, 'IfcRailing': '#95A5A6'
        }
        self.line_weights = {k: 1.0 for k in self.colors.keys()}
        self.line_weights['default'] = 1.0
        self.alphas = {k: 0.8 for k in self.colors.keys()}
        self.alphas['IfcSlab'] = 0.2
        self.alphas['default'] = 0.8
    
    def _setup_colorful_style(self):
        """Bright, colorful style for presentations"""
        self.colors = {
            'IfcWall': '#FF6B6B',           # Bright red
            'IfcWallStandardCase': '#FF6B6B',
            'IfcSlab': '#FFE66D',           # Bright yellow
            'IfcColumn': '#4ECDC4',         # Turquoise
            'IfcBeam': '#45B7D1',           # Sky blue
            'IfcDoor': '#FFA07A',           # Light salmon
            'IfcWindow': '#98D8E8',         # Light blue
            'IfcStair': '#DDA0DD',          # Plum
            'IfcStairFlight': '#DDA0DD',
            'IfcRailing': '#F0E68C',        # Khaki
            'IfcRamp': '#FFB347',           # Peach
            'IfcFurnishingElement': '#90EE90', # Light green
            'IfcSpace': '#F0F8FF',          # Alice blue
        }
        self.line_weights = {k: 1.5 for k in self.colors.keys()}
        self.line_weights['default'] = 1.5
        self.alphas = {k: 0.7 for k in self.colors.keys()}
        self.alphas['IfcSlab'] = 0.4
        self.alphas['IfcSpace'] = 0.1
        self.alphas['default'] = 0.7
    
    def _setup_technical_style(self):
        """Technical architectural drawing style - line only, no fills"""
        # All elements use black lines, no fills
        self.colors = {k: 'none' for k in [
            'IfcWall', 'IfcWallStandardCase', 'IfcSlab', 'IfcColumn', 'IfcBeam',
            'IfcDoor', 'IfcWindow', 'IfcStair', 'IfcStairFlight', 'IfcRailing',
            'IfcRamp', 'IfcFurnishingElement', 'IfcSpace', 'IfcZone'
        ]}
        
        # Different line weights for hierarchy
        self.line_weights = {
            'IfcWall': 2.0,              # Thick lines for walls
            'IfcWallStandardCase': 2.0,   
            'IfcColumn': 2.0,             # Thick for structure
            'IfcBeam': 1.5,               # Medium for beams
            'IfcSlab': 1.0,               # Medium for slabs
            'IfcDoor': 1.0,               # Medium for openings
            'IfcWindow': 1.0,
            'IfcStair': 1.5,              # Medium for stairs
            'IfcStairFlight': 1.5,
            'IfcRailing': 0.5,            # Thin for details
            'IfcFurnishingElement': 0.5,  # Thin for furniture
            'IfcSpace': 0.3,              # Very thin for spaces
            'default': 1.0
        }
        
        # No transparency for technical drawings
        self.alphas = {k: 0.0 for k in self.colors.keys()}  # 0 = no fill
        self.alphas['default'] = 0.0
        
        # Set edge colors to black for all elements
        self.edge_colors = {k: 'black' for k in self.colors.keys()}
        self.edge_colors['default'] = 'black'
    
    def process(self, name: str, elements: list, polygons: List[Tuple[str, str, Polygon]]):
        """Generate professional floor plan image"""
        
        if not polygons:
            print(f"No polygons for level {name}")
            return
        
        # Create figure with better DPI and size
        plt.style.use('default')  # Reset to clean style
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        fig.patch.set_facecolor('white')
        
        # Calculate bounds with better padding
        all_bounds = []
        for elem_type, elem_name, poly in polygons:
            if poly and not poly.is_empty:
                all_bounds.extend(poly.bounds)
        
        if all_bounds:
            min_x, min_y = min(all_bounds[::2]), min(all_bounds[1::2])
            max_x, max_y = max(all_bounds[::2]), max(all_bounds[1::2])
            
            # Better padding calculation
            range_x, range_y = max_x - min_x, max_y - min_y
            max_range = max(range_x, range_y)
            padding = max_range * 0.05  # 5% padding
            
            ax.set_xlim(min_x - padding, max_x + padding)
            ax.set_ylim(min_y - padding, max_y + padding)
        
        # Group polygons by type for better rendering order
        polygon_groups = {}
        for elem_type, elem_name, poly in polygons:
            if elem_type not in polygon_groups:
                polygon_groups[elem_type] = []
            polygon_groups[elem_type].append((elem_name, poly))
        
        # Render in specific order (background to foreground)
        render_order = [
            'IfcSpace', 'IfcZone', 'IfcSlab', 'IfcCovering',  # Background elements
            'IfcWall', 'IfcWallStandardCase', 'IfcColumn',     # Structure
            'IfcBeam', 'IfcStair', 'IfcStairFlight', 'IfcRamp', # Major elements
            'IfcDoor', 'IfcWindow',                            # Openings
            'IfcRailing', 'IfcFurnishingElement',              # Details
            'IfcFlowTerminal', 'IfcDistributionElement'        # MEP
        ]
        
        # Add any types not in the order to the end
        all_types = set(polygon_groups.keys())
        for elem_type in all_types:
            if elem_type not in render_order:
                render_order.append(elem_type)
        
        # Draw polygons in order
        for elem_type in render_order:
            if elem_type not in polygon_groups:
                continue
                
            color = self.colors.get(elem_type, '#95A5A6')  # Default gray
            alpha = self.alphas.get(elem_type, self.alphas['default'])
            line_weight = self.line_weights.get(elem_type, self.line_weights['default'])
            
            for elem_name, poly in polygon_groups[elem_type]:
                if poly is None or poly.is_empty:
                    continue
                
                # Determine edge color and fill behavior
                if hasattr(self, 'edge_colors'):  # Technical style
                    edge_color = self.edge_colors.get(elem_type, self.edge_colors['default'])
                    fill_color = 'none'  # No fill for technical drawings
                    alpha = 0.0
                else:
                    edge_color = self._darken_color(color) if elem_type != 'IfcWall' else 'black'
                    fill_color = color
                
                if isinstance(poly, Polygon):
                    self._draw_polygon(ax, poly, fill_color, alpha, edge_color, line_weight)
                elif isinstance(poly, MultiPolygon):
                    for p in poly.geoms:
                        self._draw_polygon(ax, p, fill_color, alpha, edge_color, line_weight)
        
        # Enhanced styling
        ax.set_aspect('equal')
        
        # Professional grid
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
        ax.set_axisbelow(True)
        
        # Clean, professional title
        ax.set_title(f'Floor Plan - {name}', 
                    fontsize=20, fontweight='bold', pad=20,
                    fontfamily='sans-serif')
        
        # Axis labels with units
        ax.set_xlabel('Distance (m)', fontsize=14, fontweight='medium')
        ax.set_ylabel('Distance (m)', fontsize=14, fontweight='medium')
        
        # Better tick formatting
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        # Create professional legend
        self._create_legend(ax, polygon_groups)
        
        # Add scale indicator
        self._add_scale_indicator(ax, all_bounds)
        
        # Add north arrow (if space allows)
        self._add_north_arrow(ax)
        
        # Tight layout with better spacing
        plt.tight_layout(pad=2.0)
        
        # Save with high quality
        output_path = self.context["output_dir"] / f"{name}_floor_plan.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved professional floor plan: {output_path}")
    
    def _draw_polygon(self, ax, poly, color, alpha, edge_color, line_weight):
        """Draw a single polygon with proper styling"""
        
        # Handle technical style (line-only drawing)
        if color == 'none' or alpha == 0.0:
            # Draw only the outline
            x, y = poly.exterior.xy
            ax.plot(x, y, color=edge_color, linewidth=line_weight, solid_capstyle='round')
            
            # Draw holes as lines
            for interior in poly.interiors:
                x, y = interior.xy
                ax.plot(x, y, color=edge_color, linewidth=line_weight*0.7, solid_capstyle='round')
        else:
            # Draw filled polygon (other styles)
            x, y = poly.exterior.xy
            ax.fill(x, y, color=color, alpha=alpha, edgecolor=edge_color, 
                   linewidth=line_weight, zorder=1)
            
            # Holes (if any)
            for interior in poly.interiors:
                x, y = interior.xy
                ax.fill(x, y, color='white', alpha=0.9, edgecolor=edge_color, 
                       linewidth=line_weight*0.7, zorder=2)
    
    def _darken_color(self, color_hex, factor=0.7):
        """Darken a hex color for edge rendering"""
        import matplotlib.colors as mcolors
        try:
            rgb = mcolors.hex2color(color_hex)
            darkened = tuple(c * factor for c in rgb)
            return mcolors.rgb2hex(darkened)
        except:
            return 'black'
    
    def _create_legend(self, ax, polygon_groups):
        """Create a professional legend"""
        # Only show legend for types that are actually present
        present_types = list(polygon_groups.keys())
        
        if len(present_types) > 12:  # Too many for readable legend
            return
        
        # Skip legend for technical style (line-only drawings)
        if hasattr(self, 'edge_colors'):
            return
            
        legend_elements = []
        legend_labels = []
        
        for elem_type in present_types:
            if elem_type in self.colors:
                color = self.colors[elem_type]
                alpha = self.alphas.get(elem_type, self.alphas['default'])
                
                # Create legend patch
                patch = patches.Rectangle((0,0), 1, 1, 
                                        facecolor=color, alpha=alpha,
                                        edgecolor=self._darken_color(color),
                                        linewidth=1)
                legend_elements.append(patch)
                
                # Clean up label
                clean_label = elem_type.replace('Ifc', '').replace('StandardCase', '')
                legend_labels.append(clean_label)
        
        if legend_elements:
            legend = ax.legend(legend_elements, legend_labels, 
                             loc='upper left', bbox_to_anchor=(1.02, 1),
                             frameon=True, fancybox=True, shadow=True,
                             fontsize=10, title='Elements',
                             title_fontsize=12, title_fontweight='bold')
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)
    
    def _add_scale_indicator(self, ax, bounds):
        """Add a scale indicator to the plot"""
        if not bounds:
            return
            
        # Calculate appropriate scale length
        range_x = max(bounds[::2]) - min(bounds[::2])
        
        # Choose nice round number for scale
        if range_x > 50:
            scale_length = 10
        elif range_x > 20:
            scale_length = 5
        elif range_x > 10:
            scale_length = 2
        else:
            scale_length = 1
        
        # Position in bottom right
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        scale_x = xlim[1] - (xlim[1] - xlim[0]) * 0.15
        scale_y = ylim[0] + (ylim[1] - ylim[0]) * 0.05
        
        # Draw scale line
        ax.plot([scale_x - scale_length, scale_x], [scale_y, scale_y], 
               'k-', linewidth=3, solid_capstyle='butt')
        
        # Add scale text
        ax.text(scale_x - scale_length/2, scale_y + (ylim[1] - ylim[0]) * 0.02, 
               f'{scale_length}m', ha='center', va='bottom', 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def _add_north_arrow(self, ax):
        """Add a simple north arrow"""
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Position in top right
        arrow_x = xlim[1] - (xlim[1] - xlim[0]) * 0.05
        arrow_y = ylim[1] - (ylim[1] - ylim[0]) * 0.05
        
        # Simple north arrow
        ax.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - (ylim[1] - ylim[0]) * 0.03),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                   fontsize=12, fontweight='bold', ha='center')



class FloorWKTFormatter(ModernFormatter):
    """Export floor plans as WKT (Well-Known Text) CSV files"""
    
    def process(self, name: str, elements: list, polygons: List[Tuple[str, str, Polygon]]):
        """Export polygons as WKT"""
        
        from shapely import to_wkt
        
        data = {"type": [], "name": [], "geometry": [], "area": []}
        
        for elem_type, elem_name, poly in polygons:
            if poly is None or poly.is_empty:
                continue
                
            wkt = to_wkt(poly)
            data["geometry"].append(wkt)
            data["type"].append(elem_type)
            data["name"].append(elem_name)
            data["area"].append(poly.area)
        
        if data["type"]:
            df = pd.DataFrame(data)
            output_path = self.context["output_dir"] / f"{name}_floor_plan.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved WKT data: {output_path} ({len(data['type'])} elements)")


def get_elements_and_shapes_modern(model, filter_fn=None, filter_expr=None, max_elements=None):
    """
    Modern version of get_elements_and_shapes using IFCGeometryProcessor
    """
    processor = IFCGeometryProcessor()
    
    # Get elements to process
    if filter_expr and hasattr(model, 'by_type'):
        from ifcopenshell.util.selector import filter_elements
        elements = filter_elements(model, filter_expr)
    else:
        elements = model.by_type("IfcProduct")
    
    # Filter elements
    if filter_fn:
        elements = [el for el in elements if filter_fn(el)]
    
    # Limit elements for testing large files
    if max_elements and len(elements) > max_elements:
        print(f"🔄 Limiting to first {max_elements} elements for testing")
        elements = elements[:max_elements]
    
    print(f"🔄 Processing {len(elements)} filtered elements...")
    
    # Convert to meshes with progress tracking
    valid_elements = []
    meshes = []
    failed_count = 0
    skipped_count = 0
    
    # Create progress bar
    progress_bar = tqdm(elements, desc="Converting elements", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    
    for i, element in enumerate(progress_bar):
        if element.Representation is None:
            skipped_count += 1
            continue
            
        mesh = processor.process_ifc_element(element, model)
        if mesh is not None:
            valid_elements.append(element)
            meshes.append(mesh)
        else:
            failed_count += 1
        
        # Update progress bar with statistics
        processed = i + 1
        success_rate = (len(valid_elements) / processed) * 100 if processed > 0 else 0
        progress_bar.set_postfix({
            'Valid': len(valid_elements),
            'Failed': failed_count,
            'Skipped': skipped_count,
            'Success': f'{success_rate:.1f}%'
        })
    
    print(f"\n✅ Processing complete!")
    print(f"   📊 Total processed: {len(elements)}")
    print(f"   ✅ Valid geometries: {len(valid_elements)} ({(len(valid_elements)/len(elements)*100):.1f}%)")
    print(f"   ❌ Failed conversions: {failed_count} ({(failed_count/len(elements)*100):.1f}%)")
    print(f"   ⏭️  Skipped (no representation): {skipped_count} ({(skipped_count/len(elements)*100):.1f}%)")
    
    return valid_elements, meshes


def process_using_storeys_modern(context):
    """Modern version of process_using_storeys"""
    
    model = ifcopenshell.open(context["ifc_path"])
    print("Loading and filtering elements...")
    
    elements, meshes = get_elements_and_shapes_modern(
        model, 
        filter_fn=context.get("filter_fn"),
        max_elements=context.get("max_elements")
    )
    
    print(f"Loaded {len(elements)} elements with valid geometry")
    
    processor = IFCGeometryProcessor(context['engine'])
    
    # Process each storey
    storeys = list(model.by_type("IfcBuildingStorey"))
    
    for s0, s1 in zip(storeys[:-1], storeys[1:]):
        name = s0.Name or f"Level_{s0.id()}"
        # Middle height between storeys (convert from mm to m)
        section_height = (s0.Elevation + s1.Elevation) / 2000
        
        print(f"Processing storey: {name} at height {section_height:.2f}m")
        
        # Get elements in this storey
        storey_elements = get_decomposition(s0)
        storey_element_ids = {el.id() for el in storey_elements}
        
        # Filter our loaded elements to this storey
        level_polygons = []
        
        for element, mesh in zip(elements, meshes):
            if element.id() in storey_element_ids:
                # Intersect with plane
                polygons = processor.engine.intersect_with_plane(
                    mesh,
                    plane_origin=(0, 0, section_height),
                    plane_normal=(0, 0, 1)
                )
                
                for poly in polygons:
                    level_polygons.append((
                        element.is_a(),
                        element.Name or f"{element.is_a()}_{element.id()}",
                        poly
                    ))
        
        if level_polygons:
            print(f"  Found {len(level_polygons)} intersections")
            
            # Run formatters
            for formatter in context["formatters"]:
                formatter.process(name, storey_elements, level_polygons)


def process_modern(context):
    """Modern version of process function"""
    
    model = ifcopenshell.open(context["ifc_path"])
    print("🔄 Loading and filtering elements...")
    
    elements, meshes = get_elements_and_shapes_modern(
        model, 
        filter_fn=context.get("filter_fn"),
        filter_expr=context.get("filter"),
        max_elements=context.get("max_elements")
    )
    
    print(f"🏗️  Ready to process {len(elements)} elements with valid geometry")
    
    processor = IFCGeometryProcessor(context['engine'])
    levels = context["levels"]
    
    # Process each level
    for level_idx, (name, section_height) in enumerate(levels):
        print(f"\n{'='*60}")
        print(f"🏢 Processing Level {level_idx + 1}/{len(levels)}: {name}")
        print(f"📏 Section height: {section_height:.2f}m")
        print(f"{'='*60}")
        
        level_polygons = []
        intersected_count = 0
        eligible_count = 0
        
        # Create progress bar for level processing
        progress_bar = tqdm(zip(elements, meshes), 
                           desc=f"Level: {name}", 
                           total=len(elements),
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        
        for i, (element, mesh) in enumerate(progress_bar):
            # Check if element might intersect the plane
            bounds = mesh.bounds
            tolerance = 0.1
            
            # Handle both 2D and 3D bounds
            if len(bounds) == 6:  # 3D bounds
                zmin, zmax = bounds[2], bounds[5]
            elif len(bounds) == 4:  # 2D bounds
                zmin, zmax = 0, 0
            else:
                continue
            
            if zmin <= section_height + tolerance and zmax >= section_height - tolerance:
                eligible_count += 1
                # Intersect with plane
                try:
                    polygons = processor.engine.intersect_with_plane(
                        mesh,
                        plane_origin=(0, 0, section_height),
                        plane_normal=(0, 0, 1)
                    )
                    
                    if polygons:
                        intersected_count += 1
                        for poly in polygons:
                            level_polygons.append((
                                element.is_a(),
                                element.Name or f"{element.is_a()}_{element.id()}",
                                poly
                            ))
                except Exception as e:
                    # Silently continue on errors to avoid cluttering output
                    continue
            
            # Update progress bar
            processed = i + 1
            intersection_rate = (intersected_count / eligible_count * 100) if eligible_count > 0 else 0
            progress_bar.set_postfix({
                'Eligible': eligible_count,
                'Intersected': intersected_count,
                'Polygons': len(level_polygons),
                'Hit Rate': f'{intersection_rate:.1f}%'
            })
        
        print(f"\n📊 Level {name} Results:")
        print(f"   🎯 Elements checked: {len(elements)}")
        print(f"   ✅ Eligible for intersection: {eligible_count} ({(eligible_count/len(elements)*100):.1f}%)")
        print(f"   🔀 Successfully intersected: {intersected_count} ({(intersected_count/eligible_count*100 if eligible_count > 0 else 0):.1f}%)")
        print(f"   🔺 Total polygons found: {len(level_polygons)}")
        
        if level_polygons:
            # Run formatters
            print(f"🎨 Generating outputs...")
            for formatter in context["formatters"]:
                formatter.process(name, elements, level_polygons)
            print(f"✅ Level {name} completed successfully!")
        else:
            print(f"⚠️  No intersections found for level {name}")




def default_filter():
    """Default filter function - exclude problematic element types"""
    def fn(el):
        # Skip annotations and other non-geometric elements that often cause issues
        skip_types = {
            "IfcSpace", "IfcAnnotation", "IfcGrid", "IfcGridAxis", 
            "IfcOpeningElement", "IfcVirtualElement", "IfcProjectionElement"
        }
        
        return (el.is_a("IfcProduct") and 
                el.Representation is not None and
                not any(el.is_a(skip_type) for skip_type in skip_types))
    return fn


def main():
    """Main function with modernized argument parsing"""
    
    parser = argparse.ArgumentParser(description="Extract floor plans from IFC files (modernized)")
    parser.add_argument("ifc_paths", help="IFC file path or glob pattern")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--use-storey", action="store_true", 
                       help="Use IfcBuildingStorey elements to infer floors")
    parser.add_argument("--formatter", nargs='+', default=[], 
                       choices=["image", "wkt"], help="Output formatters (space-separated list)")
    parser.add_argument("--filter", help="Filter expression for IfcOpenShell")
    parser.add_argument("--width", type=int, default=2048, help="Image width")
    parser.add_argument("--height", type=int, default=2048, help="Image height")
    parser.add_argument("--style", choices=["professional", "minimal", "colorful", "technical"], 
                       default="professional", help="Visual style theme")
    parser.add_argument("--max-elements", type=int, default=None, 
                       help="Maximum number of elements to process (for testing large files)")
    parser.add_argument("--skip-failed", action="store_true", 
                       help="Continue processing even if some elements fail")
    parser.add_argument("--tolerance", type=float, default=1e-6, 
                       help="Geometric tolerance")
    
    args = parser.parse_args()
    
    # Setup context
    context = {
        "args": args,
        "engine": ShapelyTrimeshEngine(),
        "filter_fn": default_filter(),
        "filter": args.filter,
        "style": args.style
    }
    
    # Setup formatters
    if not args.formatter:
        selected_formatters = ["image", "wkt"]
    else:
        selected_formatters = args.formatter
    
    context["formatters"] = []
    for formatter_name in selected_formatters:
        if formatter_name == "image":
            context["formatters"].append(FloorPlanImageFormatter(context))
        elif formatter_name == "wkt":
            context["formatters"].append(FloorWKTFormatter(context))
    
    # Process IFC files
    ifc_paths = glob.glob(args.ifc_paths)
    if not ifc_paths:
        print(f"No IFC files found matching: {args.ifc_paths}")
        return
    
    for ifc_path in ifc_paths:
        print(f"\n{'='*60}")
        print(f"Processing: {ifc_path}")
        print(f"{'='*60}")
        
        ifc_path = Path(ifc_path)
        output_dir = Path(args.output) / ifc_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        context["output_dir"] = output_dir
        context["ifc_path"] = ifc_path
        context["max_elements"] = args.max_elements
        
        try:
            if args.use_storey:
                process_using_storeys_modern(context)
            else:
                # Look for level file
                level_file = ifc_path.parent / f"{ifc_path.stem}.csv"
                
                if level_file.exists():
                    print(f"Using level file: {level_file}")
                    df = pd.read_csv(level_file, names=["level", "elevation"])
                    # Convert to list of (name, height_in_meters)
                    levels = []
                    for i in range(len(df) - 1):
                        level_name = df.iloc[i]["level"]
                        height = (df.iloc[i]["elevation"] + df.iloc[i+1]["elevation"]) / 2000
                        levels.append((level_name, height))
                    
                    context["levels"] = levels
                    process_modern(context)
                else:
                    print(f"Warning: No level file found at {level_file}")
                    print("Using storey-based processing instead...")
                    process_using_storeys_modern(context)
        
        except Exception as e:
            print(f"Error processing {ifc_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"Output saved to: {Path(args.output).absolute()}")


if __name__ == "__main__":
    main()
    