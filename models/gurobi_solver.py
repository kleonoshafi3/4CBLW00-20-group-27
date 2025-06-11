import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import calendar
from datetime import datetime
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class PoliceResourceSolver:
    def __init__(self, data_dir="."):
        """Initialize the solver with ward temporal data"""
        self.data_dir = data_dir
        self.load_data()
        
        # Model parameters
        self.OFFICERS_PER_WARD = 100
        self.PATROL_HOURS = list(range(6, 22))  # 06:00-21:00
        self.BURGLARY_HOURS_PER_DAY = 200
        self.PATROL_DAYS_PER_WEEK = 4
        self.MAX_SHIFT_HOURS = 12
        self.ALPHA = 0.5  # Regular patrol weight
        self.BETA = 1.0   # Special operations weight
        self.SO_CAPACITY = 50
        self.SO_DURATION = 8
        self.MAX_SO_PER_PERIOD = 1

        # Hourly risk profile based on burglary statistics
        self.hourly_risk = {}
        for h in self.PATROL_HOURS:
            if 6 <= h < 18:
                self.hourly_risk[h] = 0.4 / 12  # 40% spread over 12 hours (6am-6pm)
            else:
                self.hourly_risk[h] = 0.6 / 4   # 60% spread over 4 hours (6pm-10pm)

    def load_data(self):
        """Load and process the ward temporal data"""
        try:
            # Load the ward temporal analysis data
            df = pd.read_csv(os.path.join(self.data_dir, 'output_csv_files', 'ward_temporal_analysis.csv'))
            df['Month'] = pd.to_datetime(df['Month'])
            
            # Initialize data structures
            self.wards = sorted(df['Ward ID'].unique())
            self.months = sorted(df['Month'].unique())
            
            # Create dictionaries with default values for missing data
            self.ward_areas = {}
            self.ward_names = {}
            self.crime_density = {}
            self.avg_crime_density = {}
            
            # Process each ward
            for ward in self.wards:
                ward_data = df[df['Ward ID'] == ward]
                self.ward_areas[ward] = ward_data['Ward_Area'].iloc[0]
                self.ward_names[ward] = ward_data['Ward Name'].iloc[0]
                # Initialize crime density with zeros
                densities = []
                for month in self.months:
                    val = 0.0
                    if not ward_data[ward_data['Month'] == month].empty:
                        val = ward_data[ward_data['Month'] == month]['Monthly_Crime_Density'].iloc[0]
                    self.crime_density[ward, month] = val
                    densities.append(val)
                # Calculate average crime density for the ward
                self.avg_crime_density[ward] = sum(densities) / len(densities) if densities else 0.0
            print(f"Loaded data for {len(self.wards)} wards from {self.months[0]} to {self.months[-1]}")
            print(f"Total months: {len(self.months)}")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def calculate_min_max_officers(self, ward):
        """Calculate minimum and maximum officers per hour based on average crime density, with a wider range"""
        avg_density = self.avg_crime_density.get(ward, 0)
        max_density = max(self.avg_crime_density.values())
        normalized_density = avg_density / max_density if max_density > 0 else 0
        min_officers = max(1, int(3 + 7 * normalized_density))
        max_officers = min(self.OFFICERS_PER_WARD, int(10 + 40 * normalized_density))
        return min_officers, max_officers

    def build_model(self):
        """Build the Gurobi optimization model for a typical day"""
        model = gp.Model("PoliceResourceAllocation_Daily")
        self.x = {}  # Daily officer-hours
        self.area_cov = {}  # Area coverage fraction
        self.max_cov = {}  # Maximum coverage indicator
        self.hourly_alloc = {}  # Hourly officer allocation
        for w in self.wards:
            # Daily officer-hours
            self.x[w] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_{w}")
            self.area_cov[w] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"area_cov_{w}")
            self.max_cov[w] = model.addVar(vtype=GRB.BINARY, name=f"max_cov_{w}")
            for h in self.PATROL_HOURS:
                self.hourly_alloc[w, h] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"hourly_{w}_{h}")
        self.add_constraints(model)
        self.set_objective(model)
        model.update()
        return model

    def add_constraints(self, model):
        """Add all constraints to the model for a typical day"""
        for w in self.wards:
            min_officers, max_officers = self.calculate_min_max_officers(w)
            for h in self.PATROL_HOURS:
                model.addConstr(
                    self.hourly_alloc[w, h] >= min_officers,
                    name=f"min_officers_hour_{w}_{h}"
                )
                model.addConstr(
                    self.hourly_alloc[w, h] <= max_officers,
                    name=f"max_officers_hour_{w}_{h}"
                )
            model.addConstr(
                self.x[w] == gp.quicksum(self.hourly_alloc[w, h] for h in self.PATROL_HOURS),
                name=f"total_hours_{w}"
            )
            avg_officers = self.x[w] / len(self.PATROL_HOURS)
            model.addConstr(
                self.area_cov[w] * max_officers <= avg_officers,
                name=f"area_cov_calc_{w}"
            )
            model.addConstr(
                self.area_cov[w] >= 0.9 * self.max_cov[w],
                name=f"max_cov_ind_{w}"
            )
            # Shift coverage constraints can be added here if needed

    def set_objective(self, model):
        """Set the objective function to maximize hourly officer effectiveness with risk profile for a typical day"""
        effectiveness = gp.quicksum(
            self.avg_crime_density[w] * self.hourly_risk[h] * self.hourly_alloc[w, h]
            for w in self.wards
            for h in self.PATROL_HOURS
        )
        underutilization = gp.quicksum(
            (self.OFFICERS_PER_WARD - self.hourly_alloc[w, h]) * (1 - self.avg_crime_density[w])
            for w in self.wards
            for h in self.PATROL_HOURS
        )
        model.setObjective(effectiveness - 0.1 * underutilization, GRB.MAXIMIZE)

    def solve(self, model):
        """Solve the optimization problem"""
        # Set solver parameters
        model.setParam('TimeLimit', 600)  # 10 minutes
        model.setParam('MIPGap', 0.05)    # 5% gap
        model.setParam('OutputFlag', 1)
        
        # Solve
        model.optimize()
        
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            return self.get_solution(model)
        return None

    def get_solution(self, model):
        """Retrieve the solution from the model"""
        solution = {
            'regular_operations': {},
            'hourly_allocations': {},
            'objective_value': model.objVal
        }
        
        # Get regular operations solution
        for w in self.wards:
            solution['regular_operations'][w] = {}
            solution['hourly_allocations'][w] = {}
            
            for h in self.PATROL_HOURS:
                solution['regular_operations'][w][h] = {
                    'hours': self.x[w],
                    'area_coverage': self.area_cov[w],
                    'max_coverage': self.max_cov[w],
                    'crime_density': self.avg_crime_density[w]
                }
                
                # Get hourly allocations
                solution['hourly_allocations'][w][h] = self.hourly_alloc[w, h].X
        
        return solution

    def generate_pdf_report(self, solution, output_path):
        """Generate a PDF report for the solution"""
        try:
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            elements = []
            
            # Title
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            )
            elements.append(Paragraph("Police Resource Allocation Report", title_style))
            
            # Summary section
            elements.append(Paragraph("Summary", styles['Heading2']))
            elements.append(Paragraph(f"Total Crime Prevention Score: {solution['objective_value']:,.2f}", styles['Normal']))
            elements.append(Spacer(1, 20))
            
            # Regular operations
            elements.append(Paragraph("Regular Operations", styles['Heading2']))
            for ward in self.wards:
                ward_name = self.ward_names.get(ward, f"Ward {ward}")
                ward_area = self.ward_areas.get(ward, 0)
                elements.append(Paragraph(f"Ward {ward} ({ward_name}, Area: {ward_area:.2f} km²)", styles['Heading3']))
                
                # Create table for regular operations
                data = [['Hour', 'Officer Hours', 'Area Coverage', 'Max Coverage', 'Crime Density']]
                for h in self.PATROL_HOURS:
                    if h in solution['regular_operations'][ward]:
                        op = solution['regular_operations'][ward][h]
                        data.append([
                            f"{h:02d}:00-{h+1:02d}:00",
                            f"{op['hours']:.1f}",
                            f"{op['area_coverage']:.2%}",
                            f"{op['max_coverage']:.2f}",
                            f"{op['crime_density']:.2f}"
                        ])
                
                if len(data) > 1:
                    table = Table(data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, -1), 10),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    elements.append(table)
                    elements.append(Spacer(1, 20))
            
            # Build the PDF
            doc.build(elements)
            return output_path
            
        except Exception as e:
            print(f"Error generating PDF report: {str(e)}")
            return None

def main():
    """Main function to run the police resource optimization process"""
    try:
        # Initialize the solver
        solver = PoliceResourceSolver()
        
        # Build and solve the model
        model = solver.build_model()
        if model is None:
            print("Failed to build model")
            return
            
        solution = solver.solve(model)
        if solution is None:
            print("Failed to find solution")
            return
            
        # Generate reports
        print("\nGenerating reports...")
        
        # Generate main PDF report
        main_pdf_path = "police_resource_report.pdf"
        solver.generate_pdf_report(solution, main_pdf_path)
        print(f"\nMain PDF report generated: {os.path.abspath(main_pdf_path)}")
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
