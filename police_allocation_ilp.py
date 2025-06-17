import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import numpy as np
import argparse

class PoliceAllocationOptimizer:
    def __init__(self, crime_data_file):
        """
        Initialize the Police Allocation Optimizer with dynamic time weights based on crime density
        """
        self.crime_data = pd.read_csv('output_csv_files/ward_temporal_analysis.csv')
        self.model = None
        self.solution = {}
        
        # Time slots (2-hour shifts from 6AM to 10PM - daytime only)
        self.time_slots = ['06-08', '08-10', '10-12', '12-14', '14-16', '16-18', '18-20', '20-22']
        self.day_types = ['weekday', 'weekend']
        self.months = [1, 2]  # Only January and February 2025
        
        # Base time weights (for medium crime areas)
        self.base_time_weights = {
            '06-08': 1.0,    # Early Morning (baseline)
            '08-10': 1.2,    # Late Morning
            '10-12': 1.8,    # Early Afternoon
            '12-14': 2.4,    # Mid Afternoon
            '14-16': 3.2,    # Early Peak
            '16-18': 4.0,    # Late Peak
            '18-20': 3.0,    # Early Evening
            '20-22': 2.0     # Late Evening
        }
        
        # Day weights (weekday vs weekend)
        self.day_weights = {
            'weekday': 2.33,  # Weekday gets 70% of total allocation
            'weekend': 1.0    # Weekend gets 30% of total allocation
        }
        
        # Linear effectiveness parameter (removed quadratic terms)
        self.alpha = 25.0   # Increased linear effectiveness for pure linear model
        self.gamma = 100.0  # Special operations bonus
        
        # Process crime data
        self.crime_data['Month'] = pd.to_datetime(self.crime_data['Month'])
        self.crime_data['year'] = self.crime_data['Month'].dt.year
        self.crime_data['month'] = self.crime_data['Month'].dt.month
        self.crime_data.rename(columns={'Ward ID': 'ward_id', 'Monthly_Crime_Density': 'crime_density'}, inplace=True)
        
        # Filter data for January and February 2025 only
        self.crime_data = self.crime_data[
            (self.crime_data['year'] == 2025) & 
            (self.crime_data['month'].isin([1, 2]))
        ]
        
        self.wards = self.crime_data['ward_id'].unique()
        
        # Calculate crime density statistics for 2025 only
        self.max_crime_density = self.crime_data['crime_density'].max()
        self.min_crime_density = self.crime_data['crime_density'].min()
        self.mean_crime_density = self.crime_data['crime_density'].mean()
        self.std_crime_density = self.crime_data['crime_density'].std()
        
        print(f"\n2025 Crime Density Statistics (Jan-Feb):")
        print(f"Maximum Crime Density: {self.max_crime_density:.2f}")
        print(f"Minimum Crime Density: {self.min_crime_density:.2f}")
        print(f"Mean Crime Density: {self.mean_crime_density:.2f}")
        print(f"Standard Deviation: {self.std_crime_density:.2f}")
        
        # Improved normalization - preserve more variation
        crime_range = self.max_crime_density - self.min_crime_density
        if crime_range > 0:
            self.crime_data['normalized_crime_density'] = (
                (self.crime_data['crime_density'] - self.min_crime_density) / crime_range * 2 + 0.5
            )  # Scale to 0.5-2.5 range
        else:
            self.crime_data['normalized_crime_density'] = 1.0
        
        # Create monthly crime density lookup
        self.monthly_crime_density = {}
        for _, row in self.crime_data.iterrows():
            key = (row['ward_id'], row['month'], row['year'])
            self.monthly_crime_density[key] = row['normalized_crime_density']
        
        self.setup_model()
    
    def get_monthly_crime_density(self, ward, month, year):
        """Get normalized crime density for specific ward, month, year"""
        key = (ward, month, year)
        return self.monthly_crime_density.get(key, 1.0)  # Default to medium level
    
    def get_dynamic_time_weights(self, crime_density_normalized):
        """Get time slot weights that scale with crime density"""
        
        # Crime density scaling factor
        if crime_density_normalized > 1.8:  # High crime
            scale_factor = 1.5  # Amplify time differences more
            crime_level = 'high'
        elif crime_density_normalized < 1.2:  # Low crime  
            scale_factor = 0.8  # Reduce time differences
            crime_level = 'low'
        else:  # Medium crime
            scale_factor = 1.0  # Use base weights
            crime_level = 'medium'
        
        # Apply scaling to time weights while preserving relative ratios
        dynamic_weights = {}
        base_min = min(self.base_time_weights.values())
        for time_slot, base_weight in self.base_time_weights.items():
            # Scale relative to minimum weight to preserve ratios
            dynamic_weights[time_slot] = base_weight * scale_factor
        
        return dynamic_weights, crime_level, scale_factor
    
    def get_crime_level_multiplier(self, ward, month, year):
        """Get crime level multiplier and info"""
        monthly_crime_density = self.get_monthly_crime_density(ward, month, year)
        dynamic_weights, crime_level, scale_factor = self.get_dynamic_time_weights(monthly_crime_density)
        
        # Return multiplier based on crime density
        if crime_level == 'high':
            multiplier = 2.5
        elif crime_level == 'low':
            multiplier = 1.0
        else:
            multiplier = 1.8
            
        return multiplier, crime_level, monthly_crime_density
    
    def setup_model(self):
        """Setup the Linear Programming model with dynamic bounds and constraints"""
        self.model = gp.Model("PoliceAllocationLinear")
        
        # Decision Variables
        self.x = {}  # Officer assignments: x[w,t,d,m,y]
        self.y = {}  # Special operations: y[w,m,y]
        
        years = self.crime_data['year'].unique()
        
        # Create officer assignment variables with dynamic bounds
        for w in self.wards:
            for t in self.time_slots:
                for d in self.day_types:
                    for m in self.months:
                        for y in years:
                            # Get crime density for dynamic bounds
                            crime_density = self.get_monthly_crime_density(w, m, y)
                            min_officers, max_officers = self.get_dynamic_officer_bounds(
                                crime_density, t, d
                            )
                            
                            self.x[w, t, d, m, y] = self.model.addVar(
                                vtype=GRB.INTEGER, 
                                lb=min_officers,
                                ub=max_officers,
                                name=f"officers_{w}_{t}_{d}_{m}_{y}"
                            )
        
        # Create special operations variables
        for w in self.wards:
            for y in years:
                for m in self.months:
                    self.y[w, m, y] = self.model.addVar(
                        vtype=GRB.BINARY, 
                        name=f"special_op_{w}_{m}_{y}"
                    )
        
        self.add_improved_constraints(years)
        self.set_linear_objective(years)
        
        # Set model parameters for linear programming
        self.model.setParam('TimeLimit', 300)  # Reduced time limit for linear model
        self.model.setParam('MIPGap', 0.01)
        self.model.setParam('FeasibilityTol', 1e-6)
    
    def get_dynamic_officer_bounds(self, crime_density, time_slot, day_type):
        """Get dynamic officer bounds based on crime density"""
        
        # Base bounds by time slot and day type
        base_bounds = {
            ('weekday', '06-08'): (1, 4),    # Minimum 1 officer, flexible maximum
            ('weekday', '08-10'): (1, 6),
            ('weekday', '10-12'): (1, 8),
            ('weekday', '12-14'): (1, 10),
            ('weekday', '14-16'): (1, 12),
            ('weekday', '16-18'): (1, 12),
            ('weekday', '18-20'): (1, 10),
            ('weekday', '20-22'): (1, 8),
            ('weekend', '06-08'): (1, 3),
            ('weekend', '08-10'): (1, 4),
            ('weekend', '10-12'): (1, 5),
            ('weekend', '12-14'): (1, 6),
            ('weekend', '14-16'): (1, 8),
            ('weekend', '16-18'): (1, 8),
            ('weekend', '18-20'): (1, 6),
            ('weekend', '20-22'): (1, 5)
        }
        
        min_base, max_base = base_bounds[(day_type, time_slot)]
        
        # Scale bounds based on crime density
        if crime_density > 1.8:  # High crime
            min_officers = max(1, int(min_base * 1.2))  # Ensure minimum of 1
            max_officers = int(max_base * 1.5)  # More officers for high crime
        elif crime_density < 1.2:  # Low crime
            min_officers = 1  # Always minimum 1 officer
            max_officers = int(max_base * 0.8)
        else:  # Medium crime
            min_officers = max(1, min_base)  # Ensure minimum of 1
            max_officers = int(max_base * 1.2)
        
        return min_officers, max_officers
    
    def add_improved_constraints(self, years):
        """Add improved constraints with daily caps"""
        
        # Daily officer and hours constraints with your specified caps
        for w in self.wards:
            for y in years:
                for m in self.months:
                    for d in self.day_types:
                        # Daily officer cap: 100 officers per day per ward
                        self.model.addConstr(
                            gp.quicksum(self.x[w, t, d, m, y] for t in self.time_slots) <= 100,
                            name=f"daily_officer_cap_{w}_{d}_{m}_{y}"
                        )
                        
                        # Daily hours cap: 200 officer-hours per day per ward
                        self.model.addConstr(
                            gp.quicksum(self.x[w, t, d, m, y] * 2 for t in self.time_slots) <= 200,
                            name=f"daily_hours_cap_{w}_{d}_{m}_{y}"
                        )
                        
                        # Ensure minimum coverage in each time slot
                        for t in self.time_slots:
                            self.model.addConstr(
                                self.x[w, t, d, m, y] >= 1,
                                name=f"min_coverage_{w}_{t}_{d}_{m}_{y}"
                            )
        
        # Time slot priority constraints (more flexible)
        for w in self.wards:
            for y in years:
                for m in self.months:
                    for d in self.day_types:
                        # Get crime density for this ward/month/year
                        crime_density = self.get_monthly_crime_density(w, m, y)
                        dynamic_weights, crime_level, scale_factor = self.get_dynamic_time_weights(crime_density)
                        
                        # Calculate relative weights for peak times
                        weight_16_18 = dynamic_weights['16-18'] / dynamic_weights['06-08']
                        weight_14_16 = dynamic_weights['14-16'] / dynamic_weights['06-08']
                        weight_18_20 = dynamic_weights['18-20'] / dynamic_weights['06-08']
                        
                        # Add flexible constraints based on weights
                        self.model.addConstr(
                            self.x[w, '16-18', d, m, y] >= weight_16_18 * 0.8 * self.x[w, '06-08', d, m, y],
                            name=f"peak_priority_1_{w}_{d}_{m}_{y}"
                        )
                        self.model.addConstr(
                            self.x[w, '14-16', d, m, y] >= weight_14_16 * 0.8 * self.x[w, '06-08', d, m, y],
                            name=f"peak_priority_2_{w}_{d}_{m}_{y}"
                        )
                        self.model.addConstr(
                            self.x[w, '18-20', d, m, y] >= weight_18_20 * 0.8 * self.x[w, '06-08', d, m, y],
                            name=f"peak_priority_3_{w}_{d}_{m}_{y}"
                        )
        
        # Weekday vs Weekend constraints
        for w in self.wards:
            for y in years:
                for m in self.months:
                    # Total weekday allocation should be higher than weekend
                    weekday_total = gp.quicksum(self.x[w, t, 'weekday', m, y] for t in self.time_slots)
                    weekend_total = gp.quicksum(self.x[w, t, 'weekend', m, y] for t in self.time_slots)
                    
                    self.model.addConstr(
                        weekday_total >= 2.0 * weekend_total,  # Weekday gets at least 2x more officers
                        name=f"weekday_priority_{w}_{m}_{y}"
                    )
        
        # Special operations constraints
        for w in self.wards:
            for y in years:
                max_special_ops = 6  # Increased flexibility
                
                self.model.addConstr(
                    gp.quicksum(self.y[w, m, y] for m in self.months) <= max_special_ops,
                    name=f"special_ops_limit_{w}_{y}"
                )
    
    def set_linear_objective(self, years):
        """Set LINEAR objective function with dynamic time weights"""
        objective = 0
        
        # Main allocation objective with dynamic time weights
        for w in self.wards:
            for t in self.time_slots:
                for d in self.day_types:
                    for m in self.months:
                        for y in years:
                            # Get crime density and dynamic weights
                            crime_density = self.get_monthly_crime_density(w, m, y)
                            dynamic_time_weights, crime_level, scale_factor = self.get_dynamic_time_weights(crime_density)
                            
                            # Calculate weight factor incorporating crime density and day weights
                            weight_factor = (dynamic_time_weights[t] * 
                                           self.day_weights[d] * 
                                           crime_density *  # Include crime density for scaling
                                           self.alpha)
                            
                            # PURE LINEAR OBJECTIVE (no quadratic terms)
                            linear_term = weight_factor * self.x[w, t, d, m, y]
                            objective += linear_term
        
        # Special operations bonus
        for w in self.wards:
            for y in years:
                for m in self.months:
                    crime_density = self.get_monthly_crime_density(w, m, y)
                    special_bonus = self.gamma * crime_density
                    objective += special_bonus * self.y[w, m, y]
        
        self.model.setObjective(objective, GRB.MAXIMIZE)
    
    def optimize(self):
        """Solve the linear optimization problem"""
        print("Starting LINEAR optimization with dynamic time weights...")
        try:
            self.model.optimize()
            
            if self.model.status == GRB.OPTIMAL:
                print(f"Optimal solution found!")
                print(f"Objective value: {self.model.objVal:.2f}")
                self.extract_solution()
                self.analyze_dynamic_solution()
                return True
            elif self.model.status == GRB.INFEASIBLE:
                print("Model is infeasible. Computing IIS...")
                self.model.computeIIS()
                print("Infeasible constraints:")
                for c in self.model.getConstrs():
                    if c.IISConstr:
                        print(f"  - {c.ConstrName}")
                return False
            elif self.model.status == GRB.TIME_LIMIT:
                print("Time limit reached. Using best solution found...")
                if self.model.solCount > 0:
                    self.extract_solution()
                    self.analyze_dynamic_solution()
                    return True
                else:
                    print("No feasible solution found within time limit.")
                    return False
            else:
                print(f"Optimization failed with status: {self.model.status}")
                return False
                
        except gp.GurobiError as e:
            print(f"Gurobi error occurred: {str(e)}")
            return False
        except Exception as e:
            print(f"Unexpected error occurred: {str(e)}")
            return False
    
    def extract_solution(self):
        """Extract the optimal solution"""
        self.solution = {
            'officer_assignments': {},
            'special_operations': {},
            'ward_totals': {},
            'time_slot_analysis': {}
        }
        
        # Extract officer assignments
        for w in self.wards:
            self.solution['officer_assignments'][w] = {}
            ward_total = 0
            
            for t in self.time_slots:
                self.solution['officer_assignments'][w][t] = {}
                for d in self.day_types:
                    self.solution['officer_assignments'][w][t][d] = {}
                    for m in self.months:
                        self.solution['officer_assignments'][w][t][d][m] = {}
                        for y in self.crime_data['year'].unique():
                            if self.x[w, t, d, m, y].x > 0.5:
                                officers = int(round(self.x[w, t, d, m, y].x))
                                self.solution['officer_assignments'][w][t][d][m][y] = officers
                                ward_total += officers
            
            self.solution['ward_totals'][w] = ward_total
        
        # Extract special operations
        for w in self.wards:
            self.solution['special_operations'][w] = {}
            for y in self.crime_data['year'].unique():
                self.solution['special_operations'][w][y] = []
                for m in self.months:
                    if self.y[w, m, y].x > 0.5:
                        self.solution['special_operations'][w][y].append(m)
    
    def analyze_dynamic_solution(self):
        """Analyze the solution showing dynamic time weight effects"""
        print("\n=== DYNAMIC TIME WEIGHTS ANALYSIS ===")
        
        # Analyze different crime density areas
        sample_wards = self.wards[:3]  # Take first 3 wards as examples
        sample_month = 1
        sample_year = 2025  # Changed to 2025
        
        for ward in sample_wards:
            crime_density = self.get_monthly_crime_density(ward, sample_month, sample_year)
            dynamic_weights, crime_level, scale_factor = self.get_dynamic_time_weights(crime_density)
            
            print(f"\n--- Ward {ward} Analysis ---")
            print(f"Crime Density: {crime_density:.3f} ({crime_level.upper()})")
            print(f"Time Weight Scale Factor: {scale_factor:.2f}")
            
            print("\nDynamic Time Weights:")
            for time_slot, weight in dynamic_weights.items():
                base_weight = self.base_time_weights[time_slot]
                print(f"  {time_slot}: {weight:.3f} (base: {base_weight:.3f})")
            
            print("\nActual Allocation:")
            weekday_total = 0
            weekend_total = 0
            
            for d in self.day_types:
                print(f"  {d.capitalize()}:")
                for t in self.time_slots:
                    if (ward, t, d, sample_month, sample_year) in self.x:
                        officers = int(round(self.x[ward, t, d, sample_month, sample_year].x))
                        if d == 'weekday':
                            weekday_total += officers
                        else:
                            weekend_total += officers
                        print(f"    {t}: {officers:2d} officers")
            
            print(f"  Daily Totals: Weekday={weekday_total}, Weekend={weekend_total}")
            if weekend_total > 0:
                print(f"  Weekday/Weekend Ratio: {weekday_total/weekend_total:.2f}")
            else:
                print("  Weekday/Weekend Ratio: N/A")
    
    

    def generate_pdf_report(self, ward_id, month):
        """Generate a PDF report for the optimization results
        
        Args:
            ward_id: Ward ID for the report
            month: Month (1 for January, 2 for February)
        """
        # Validate inputs
        if month not in [1, 2]:
            raise ValueError("Month must be 1 (January) or 2 (February)")
        
        if ward_id not in self.wards:
            raise ValueError(f"Ward {ward_id} not found in data")
        
        # Constants
        year = 2025
        
        # Create reports directory if it doesn't exist
        os.makedirs('models/reports', exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        month_name = "January" if month == 1 else "February"
        pdf_path = f'models/reports/ward_{ward_id}_report_2025_{month:02d}_{timestamp}.pdf'
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        elements.append(Paragraph(f"Police Officer Allocation Report", title_style))
        
        # Introduction
        intro_style = ParagraphStyle(
            'Introduction',
            parent=styles['Normal'],
            fontSize=12,
            leading=16,
            spaceAfter=20
        )
        intro_text = """
        This optimization system strategically deploys police officers based on actual crime data rather than equal distribution. 
        It ensures smart resource management by positioning officers where they prevent the most crime while protecting taxpayer resources. 
        The system makes evidence-based decisions that remove guesswork, and maintains officer wellbeing by avoiding unnecessary 
        over-deployment in low-crime areas that would cause fatigue and reduce operational effectiveness.
        """
        elements.append(Paragraph(intro_text, intro_style))
        elements.append(Spacer(1, 20))
        
        # Ward and Period Information
        elements.append(Paragraph(f"Ward: {ward_id}", styles['Heading2']))
        elements.append(Paragraph(f"Period: 2025-{month:02d}-01 ({month_name} 2025)", styles['Heading2']))
        elements.append(Spacer(1, 20))
        
        # Crime Density Analysis
        crime_density = self.get_monthly_crime_density(ward_id, month, year)
        dynamic_weights, crime_level, scale_factor = self.get_dynamic_time_weights(crime_density)
        
        elements.append(Paragraph("Crime Density Analysis", styles['Heading2']))
        elements.append(Paragraph(f"Crime Density: {crime_density:.2f}", styles['Normal']))
        elements.append(Paragraph(f"Crime Level: {crime_level.upper()}", styles['Normal']))
        elements.append(Paragraph(f"Time Weight Scaling: {scale_factor:.2f}x", styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Officer Allocation
        elements.append(Paragraph("Officer Allocation by Time Slot", styles['Heading2']))
        
        # Create allocation table
        data = [['Time Slot', 'Weekday Officers', 'Weekend Officers']]
        for t in self.time_slots:
            weekday_officers = int(round(self.x[ward_id, t, 'weekday', month, year].x))
            weekend_officers = int(round(self.x[ward_id, t, 'weekend', month, year].x))
            data.append([t, str(weekday_officers), str(weekend_officers)])
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))
        
        # Recommendations
        elements.append(Paragraph("Key Recommendations", styles['Heading2']))
        recommendations = [
            "1. Focus additional resources on high-crime density wards",
            "2. Maintain higher staffing during 14-16 time slot (early peak crime period) and 16-18 time slot (late peak crime period)",
            "3. Monitor and adjust based on actual crime patterns",
        ]
        for rec in recommendations:
            elements.append(Paragraph(rec, styles['Normal']))
        
        # Build PDF
        doc.build(elements)
        return pdf_path

# Example usage:
if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Police Resource Allocation Optimization')
    parser.add_argument('--ward_id', type=str, default='E05014072',
                      help='Ward ID for report generation (default: E05014072)')
    parser.add_argument('--month', type=int, default=1,
                      help='Month for report generation (1 for January, 2 for February 2025)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate month (only January or February 2025)
    if args.month not in [1, 2]:
        print("Error: Month must be 1 (January) or 2 (February) for 2025")
        exit(1)
    
    print(f"\n=== Selected Parameters ===")
    print(f"Ward ID: {args.ward_id}")
    print(f"Period: 2025-{args.month:02d}-01 ({'January' if args.month == 1 else 'February'} 2025)")
    print("=" * 30)
    
    # Initialize optimizer
    optimizer = PoliceAllocationOptimizer('output_csv_files/ward_temporal_analysis.csv')
    
    # Run optimization
    if optimizer.optimize():
        # Generate report for specified ward and month
        print("\n" + "="*80)
        print(f"GENERATING REPORT FOR WARD {args.ward_id} - 2025-{args.month:02d}-01")
        print("="*80)
        
        # Generate PDF report with simplified function call
        try:
            pdf_path = optimizer.generate_pdf_report(args.ward_id, month=args.month)
            print(f"\n✅ Report generated successfully!")
            print(f"📄 PDF report saved to: {pdf_path}")
        except ValueError as e:
            print(f"\n❌ Error generating report: {e}")
            exit(1)
    else:
        print("\n❌ Optimization failed. Please check the constraints and data.")

# Alternative direct usage examples:

optimizer.generate_pdf_report('E05014079', month=1)

optimizer.generate_pdf_report('E05009312', month=2)

optimizer.generate_pdf_report('E05009312', month=1)

optimizer.generate_pdf_report('E05009305', month=2)



