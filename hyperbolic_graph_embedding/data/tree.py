import os
import random
import sympy as sp
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Tuple, Dict, Any


class ExpressionGenerator:
    """Generate random symbolic expressions using SymPy."""
    
    def __init__(self, 
                 variables: List[str] = None, 
                 max_depth: int = 4, 
                 functions: List[str] = None, 
                 constants: List[float] = None
                 ):
        """
        Initialize the expression generator with customizable parameters.
        
        Args:
            variables: List of variable names to use in expressions
            max_depth: Maximum depth of the expression tree
            functions: List of function names to use (from sympy)
            constants: List of constants to include in expressions
        """
        self.variables = variables or ['x', 'y', 'z']
        self.max_depth = max_depth
        self.functions = functions or ['sin', 'cos', 'exp', 'log']
        self.constants = constants or [1, 2, 3, sp.pi, sp.E]
        
        # Create SymPy symbols for the variables
        self.symbols = {v: sp.Symbol(v) for v in self.variables}
        
        # Map function names to their SymPy implementations
        self.function_map = {
            'sin': sp.sin,
            'cos': sp.cos,
            'exp': sp.exp,
            'log': sp.log,
            'sqrt': sp.sqrt,
            'tan': sp.tan
        }
        
    def generate_expression(self, depth: int = None) -> sp.Expr:
        """
        Generate a random expression with specified maximum depth.
        
        Args:
            depth: Current depth of recursion (None for initial call)
            
        Returns:
            A SymPy expression
        """
        if depth is None:
            depth = self.max_depth
            
        if depth <= 1:
            # Base case: return a variable or constant
            choices = [
                lambda: random.choice([self.symbols[v] for v in self.variables]),
                lambda: random.choice(self.constants)
            ]
            return random.choice(choices)()
        
        # Decide what type of expression to generate
        choices = [
            # Binary operations
            lambda: self.generate_expression(depth - 1) + self.generate_expression(depth - 1),
            lambda: self.generate_expression(depth - 1) - self.generate_expression(depth - 1),
            lambda: self.generate_expression(depth - 1) * self.generate_expression(depth - 1),
            lambda: self.generate_expression(depth - 1) / self.generate_expression(depth - 1),
            # Unary functions
            lambda: self._apply_random_function(self.generate_expression(depth - 1))
        ]
        
        return random.choice(choices)()
    
    def _apply_random_function(self, expr: sp.Expr) -> sp.Expr:
        """Apply a random function from the available functions."""
        func_name = random.choice(self.functions)
        if func_name in self.function_map:
            return self.function_map[func_name](expr)
        return expr


class ExpressionVisualizer:
    """Convert SymPy expressions to NetworkX graphs and visualize them."""
    
    def __init__(self):
        """Initialize the visualizer."""
        pass
    
    def expression_to_graph(self, expr: sp.Expr) -> nx.DiGraph:
        """
        Convert a SymPy expression to a NetworkX directed graph.
        
        Args:
            expr: A SymPy expression
            
        Returns:
            A NetworkX DiGraph representing the expression
        """
        G = nx.DiGraph()
        self._build_graph(expr, G)
        return G
    
    def _build_graph(self, expr: sp.Expr, G: nx.DiGraph, parent_id: str = None) -> str:
        """Recursively build a graph from an expression."""
        # Generate a unique ID for this node
        node_id = f"node_{len(G.nodes)}"
        
        # Handle different expression types
        if isinstance(expr, sp.Symbol):
            # Variable node
            G.add_node(node_id, label=str(expr), type="variable")
        elif isinstance(expr, (sp.Integer, sp.Float, sp.Rational)) or expr in (sp.pi, sp.E):
            # Pure constant node
            G.add_node(node_id, label=str(expr), type="constant")
        elif expr.is_Function:
            # Function node (sin, cos, exp, etc.)
            func_name = type(expr).__name__
            G.add_node(node_id, label=func_name, type="function")
            
            # Process all arguments
            for arg in expr.args:
                child_id = self._build_graph(arg, G, node_id)
                G.add_edge(node_id, child_id)
        elif expr.is_Pow:
            base, exp = expr.args
            if exp.is_number and float(exp) == -1:
                # Division case: 1/x
                G.add_node(node_id, label="÷", type="operation")
                
                # Add 1 as numerator
                one_id = f"node_{len(G.nodes)}"
                G.add_node(one_id, label="1", type="constant")
                G.add_edge(node_id, one_id)
                
                # Add denominator
                denom_id = self._build_graph(base, G, node_id)
                G.add_edge(node_id, denom_id)
            else:
                # Regular power
                G.add_node(node_id, label="^", type="operation")
                base_id = self._build_graph(base, G, node_id)
                exp_id = self._build_graph(exp, G, node_id)
                G.add_edge(node_id, base_id)
                G.add_edge(node_id, exp_id)
        elif expr.is_Add:
            # Handle Addition
            G.add_node(node_id, label="+", type="operation")
            for arg in expr.args:
                child_id = self._build_graph(arg, G, node_id)
                G.add_edge(node_id, child_id)
        elif expr.is_Mul:
            # Multiplication/Division
            neg_powers = [arg for arg in expr.args if arg.is_Pow and 
                        arg.args[1].is_number and float(arg.args[1]) < 0]
            
            if neg_powers:
                # Division case
                G.add_node(node_id, label="÷", type="operation")
                
                # Group terms for numerator and denominator
                num_terms = [arg for arg in expr.args if arg not in neg_powers]
                if not num_terms:
                    # If no numerator terms, use 1
                    num_id = f"node_{len(G.nodes)}"
                    G.add_node(num_id, label="1", type="constant")
                    G.add_edge(node_id, num_id)
                elif len(num_terms) == 1:
                    # Single term
                    num_id = self._build_graph(num_terms[0], G, node_id)
                    G.add_edge(node_id, num_id)
                else:
                    # Multiple terms to multiply
                    num_expr = sp.Mul(*num_terms)
                    num_id = self._build_graph(num_expr, G, node_id)
                    G.add_edge(node_id, num_id)
                
                # Create denominator from negative powers
                if len(neg_powers) == 1:
                    # Single denominator term
                    base, exp = neg_powers[0].args
                    denom_id = self._build_graph(base**abs(exp), G, node_id)
                    G.add_edge(node_id, denom_id)
                else:
                    # Multiple denominator terms
                    denom_expr = sp.Mul(*[arg.args[0]**abs(arg.args[1]) for arg in neg_powers])
                    denom_id = self._build_graph(denom_expr, G, node_id)
                    G.add_edge(node_id, denom_id)
            else:
                # Regular multiplication
                G.add_node(node_id, label="×", type="operation")
                for arg in expr.args:
                    child_id = self._build_graph(arg, G, node_id)
                    G.add_edge(node_id, child_id)
        elif hasattr(expr, 'args') and expr.args:
            # Other operations with arguments
            op_name = type(expr).__name__
            G.add_node(node_id, label=op_name, type="operation")
            for arg in expr.args:
                child_id = self._build_graph(arg, G, node_id)
                G.add_edge(node_id, child_id)
        else:
            # Fallback for any other expression
            G.add_node(node_id, label=str(expr), type="other")
        
        # Connect to parent if provided
        if parent_id is not None:
            G.add_edge(parent_id, node_id)
            
        return node_id
    
    def visualize(self, G: nx.DiGraph, title: str = "Expression Tree", 
                  figsize: Tuple[int, int] = (10, 8), 
                  save_path: str = None, show: bool = True) -> None:
        """
        Visualize the expression graph.
        
        Args:
            G: NetworkX graph to visualize
            title: Plot title
            figsize: Figure size (width, height)
            save_path: Path to save the visualization (if None, don't save)
            show: Whether to display the plot
        """
        plt.figure(figsize=figsize)
        
        # Get node attributes for visualization
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        
        # Extract node labels and types
        labels = {node: data.get('label', str(node)) for node, data in G.nodes(data=True)}
        node_types = [data.get('type', 'other') for _, data in G.nodes(data=True)]
        
        # Define colors for different node types
        color_map = {
            'variable': 'skyblue',
            'constant': 'lightgreen',
            'function': 'coral',
            'operation': 'gold',
            'other': 'lightgray'
        }
        
        node_colors = [color_map.get(t, 'lightgray') for t in node_types]
        
        # Draw the graph
        nx.draw(G, pos, with_labels=True, labels=labels, node_color=node_colors, 
                node_size=2500, font_size=10, arrows=True, 
                arrowstyle='->', arrowsize=15)
        
        plt.title(title)
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Tree visualization saved to {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()


def generate_and_visualize(max_depth: int = 3, variables: List[str] = None, 
                          save_path: str = None, show: bool = True) -> Tuple[sp.Expr, nx.DiGraph]:
    """
    Generate a random expression and visualize it.
    
    Args:
        max_depth: Maximum depth of the expression tree
        variables: List of variable names to use
        save_path: Path to save the visualization (if None, don't save)
        show: Whether to display the plot
        
    Returns:
        The generated expression and corresponding graph
    """
    # Generate random expression
    generator = ExpressionGenerator(variables=variables, max_depth=max_depth)
    expr = generator.generate_expression()
    
    # Display the expression
    print("Generated Expression:")
    print(expr)
    print("\nSimplified:")
    print(sp.simplify(expr))
    
    # Convert to graph and visualize
    visualizer = ExpressionVisualizer()
    graph = visualizer.expression_to_graph(expr)
    visualizer.visualize(graph, save_path=save_path, show=show)
    
    return expr, graph

if __name__ == "__main__":
    # Example usage
    expr, graph = generate_and_visualize(max_depth=6, variables=['x', 'y'])